import builtins
import warnings
from dataclasses import dataclass
from typing import Generic, Iterable, Optional, Tuple, Type, TypeVar, List, Callable, Union
from collections import defaultdict
import csv

import yaml

warnings.filterwarnings(action="ignore")
from torch.utils.tensorboard.writer import SummaryWriter

warnings.filterwarnings(action="default")
import inspect
import os
import random
import sys
from contextlib import contextmanager
from io import TextIOWrapper

import numpy as np
import torch
import torch.backends.cudnn as cudnn

T = TypeVar("T")

_random_seed = None

def seed(value: int):
    global _random_seed
    _random_seed = value
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    np.random.seed(value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(value)

class LoggerException(Exception):
    pass

class logger:
    @dataclass(frozen=True)
    class __log_file:
        tb_logger: SummaryWriter
        log_msg_file: TextIOWrapper
        
        def close(self):
            self.tb_logger.flush()
            self.tb_logger.close()
            self.log_msg_file.close()
            
    _enabled = False
    _LOG_BASE_DIR: str = "results"
    _log_dir: Optional[str] = None
    _log_file: Optional[__log_file] = None
    
    @classmethod
    def enabled(cls) -> bool:
        return cls._enabled
            
    @classmethod
    def enable(cls, id: str, enable_log_file: bool = True):
        if not cls._enabled:
            cls._enabled = True
            cls._log_dir = f"{cls._LOG_BASE_DIR}/{id}"
            if enable_log_file:
                cls._log_file = logger.__log_file(
                    tb_logger=SummaryWriter(log_dir=cls._log_dir),
                    log_msg_file=open(f"{cls._log_dir}/log.txt", "a"),
                )
        else:
            raise LoggerException("logger is already enabled")
        
    @classmethod
    def disable(cls):
        if cls._enabled:
            if cls._log_file is not None:
                cls._log_file.log_msg_file.write("\n")
                cls._log_file.close()
                cls._log_file = None
            cls._log_dir = None
            cls._enabled = False
        else:
            raise LoggerException("logger is already disabled")
        
    @classmethod
    def print(cls, message: str, prefix: str = "[SELFIES-RND] "):
        builtins.print(f"{prefix}{message}")
        if cls._log_file is not None:
            cls._log_file.log_msg_file.write(f"{prefix}{message}\n")
            cls._log_file.log_msg_file.flush()
            
    @classmethod
    def log_data(cls, key, value, t):
        if cls._log_file is None:
            raise LoggerException("you need to enable the logger with enable_log_file option")
        cls._log_file.tb_logger.add_scalar(key, value, t)
        
    @classmethod
    def dir(cls) -> str:
        if cls._log_dir is None:
            raise LoggerException("logger is not enabled")
        return cls._log_dir

class TextInfoBox:
    def __init__(self, right_margin: int = 10) -> None:
        self._texts = []
        self._right_margin = right_margin
        self._max_text_len = 0
        
    def add_text(self, text: Optional[str]) -> "TextInfoBox":
        if text is None:
            return self
        self._max_text_len = max(self._max_text_len, len(text))
        self._texts.append((f" {text} ", " "))
        return self
        
    def add_line(self, marker: str = "-") -> "TextInfoBox":
        if len(marker) != 1:
            raise ValueError(f"marker must be one character, but {marker}")
        self._texts.append(("", marker))
        return self
    
    def make(self) -> str:
        text_info_box = f"+{self._horizontal_line()}+\n"
        for text, marker in self._texts:
            text_info_box += f"|{text}{marker * (self._max_space_len - len(text))}|\n"
        text_info_box += f"+{self._horizontal_line()}+"
        return text_info_box
    
    def _horizontal_line(self, marker: str = "-") -> str:
        return marker * (self._max_space_len)

    @property
    def _max_space_len(self) -> int:
        return self._max_text_len + self._right_margin
    
def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def save_yaml(file_path: str, data: dict):
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
        
def dict_from_keys(d: dict, keys: Iterable) -> dict:
    matched_dict = dict()
    dict_keys = d.keys()
    for key in keys:
        if key in dict_keys:
            matched_dict[key] = d[key]
    return matched_dict

def instance_from_dict(class_type: Type[T], d: dict) -> T:
    params = tuple(inspect.signature(class_type).parameters)
    param_dict = dict_from_keys(d, params)
    return class_type(**param_dict)

def exists_dir(directory) -> bool:
    return os.path.exists(directory)

def file_exists(file_path: str) -> bool:
    return os.path.isfile(file_path)

def try_create_dir(directory):
    """If there's no directory, create it."""
    try:
        if not exists_dir(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

class ItemUpdateFollower(Generic[T]):
    def __init__(self, init_item: T, include_init: bool = True):
        self._item = init_item
        self._items = []
        if include_init:
            self._items.append(init_item)
        
    def update(self, item: T):
        self._item = item
        self._items.append(item)
        
    def popall(self) -> Tuple[T, ...]:
        items = tuple(self._items)
        self._items.clear()
        return items
    
    @property
    def item(self) -> T:
        return self._item
    
    def __len__(self) -> int:
        return len(self._items)

def moving_average(values: np.ndarray, n: Optional[int] = None, smooth: Optional[float] = None):
    if (n is None and smooth is None) or (n is not None and smooth is not None):
        raise ValueError("you must specify either n or smooth")
    if smooth is not None:
        if smooth < 0.0 or smooth > 1.0:
            raise ValueError(f"smooth must be in [0, 1], but got {smooth}")
        n = int((1.0 - smooth) * 1 + smooth * len(values))
    ret = np.cumsum(values, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    ret[:n] = ret[:n] / np.arange(1, n + 1)
    return ret

def exponential_moving_average(values, smooth: float) -> np.ndarray:
    if smooth < 0.0 or smooth > 1.0:
        raise ValueError(f"smooth must be in [0, 1], but got {smooth}")
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = smooth * values[i] + (1.0 - smooth) * ema[i - 1]
    return ema


class SyncFixedBuffer(Generic[T]):
    def __init__(self, max_size: int, callback: Optional[Callable[[Iterable[T]], None]] = None):
        self._max_size = max_size
        self._buffer: List[Optional[T]] = [None for _ in range(self._max_size)]
        self._updated = [False for _ in range(self._max_size)]
        self._sync_count = 0
        self._callback = callback
        
    @property
    def sync_done(self) -> bool:
        return self._sync_count == self._max_size
        
    def __len__(self):
        return len(self._buffer)
    
    def __getitem__(self, index) -> Optional[T]:
        return self._buffer[index]
    
    def __setitem__(self, index, value: T):
        self._buffer[index] = value # type: ignore
        if not self._updated[index]:
            self._updated[index] = True
            self._sync_count += 1
        if self._callback is not None and self.sync_done:
            self._callback(tuple(self._buffer)) # type: ignore
        
    def __iter__(self):
        return iter(self._buffer)

class CSVSyncWriter:
    """
    Write a csv file with key and value fields. The key fields are used to identify the data.
    """
    def __init__(
        self,
        file_path: str,
        key_fields: Iterable[str],
        value_fields: Iterable[str],
    ) -> None:
        self._key_fields = tuple(key_fields)
        self._value_fields = tuple(value_fields)        
        self._check_fields_unique()
        self._value_buffer = defaultdict(dict)
        self._field_types = {}
        
        self._file_path = file_path
        try:
            with open(self._file_path, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError
                if len(reader.fieldnames) != len(self.fields):
                    raise FileExistsError(f"The number of fields in the csv file is different from the number of fields in the config. Create a new csv file.")
        except (FileNotFoundError, ValueError):
            # if the file does not exist or the file has no header, create a new csv file
            self._reset_csv()
                
    def add(self, keys: Union[Tuple, dict], values: dict):
        """
        Add a new data to the csv file. If the data has all required values, write it to the csv file.
        
        Args:
            keys (tuple | dict): keys of the data. You must specify all keys.
            values (dict): values of the data. It automatically extracts required values from the `values` dict.
        """
        if len(keys) != len(self._key_fields):
            raise ValueError(f"keys must have {len(self._key_fields)} elements, but got {len(keys)}")
        if isinstance(keys, dict):
            keys = tuple(keys[key_field] for key_field in self._key_fields)
        # update the buffer with the new data only if values fields is in value_fields
        self._value_buffer[keys].update(dict_from_keys(values, self._value_fields))
        # check if it has all required values for these keys
        if len(self._value_buffer[keys]) == len(self._value_fields):
            if len(self._field_types) != len(self.fields):
                key_field_types = {key_field: type(key) for key_field, key in zip(self._key_fields, keys)}
                value_field_types = {value_field: type(value) for value_field, value in self._value_buffer[keys].items() if value is not None}
                self._field_types.update(key_field_types)
                self._field_types.update(value_field_types)
            self._write_csv(keys)
            # remove the keys from the buffer
            del self._value_buffer[keys]
            
    @property
    def key_fields(self) -> Tuple[str, ...]:
        return self._key_fields
    
    @property
    def value_fields(self) -> Tuple[str, ...]:
        return self._value_fields
    
    @value_fields.setter
    def value_fields(self, value: Iterable[str]):
        # update the value fields
        self._value_fields = tuple(value)
        self._check_fields_unique()
        # update the buffer from the old csv file
        with open(self._file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys = tuple(self._field_types[key_field](row[key_field]) for key_field in self._key_fields)
                raw_value_dict = dict_from_keys(row, self._value_fields)
                # type conversion
                value_dict = {}
                for value_field, raw_value in raw_value_dict.items():
                    if raw_value is None or raw_value == "":
                        value_dict[value_field] = raw_value
                    else:
                        value_dict[value_field] = self._field_types[value_field](raw_value)                
                self._value_buffer[keys] = value_dict
        self._reset_csv()
    
    @property
    def fields(self) -> Tuple[str, ...]:
        return self.key_fields + self.value_fields
    
    def _check_fields_unique(self):
        if len(self.fields) != len(set(self.fields)):
            raise ValueError(f"all key and value fields must be unique")
    
    def _write_csv(self, keys: Tuple):
        with open(self._file_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writerow({**dict(zip(self._key_fields, keys)), **self._value_buffer[keys]})
            
    def _reset_csv(self):
        with open(self._file_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            writer.writeheader()