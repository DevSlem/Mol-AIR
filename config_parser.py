from typing import Type, TypeVar

from util import instance_from_dict, load_yaml

T = TypeVar("T")
    
class ConfigParser:
    def __init__(self, config_dict: dict) -> None:
        self._id = tuple(config_dict.keys())[0]
        self._config_dict = config_dict
        
    @property
    def id(self) -> str:
        return self._id
    
    @property
    def config_dict(self) -> dict:
        return self._config_dict
        
    def parse_train_config(self, class_type: Type[T]) -> T:
        return instance_from_dict(class_type, self._config_dict[self._id]["Train"])
    
    def parse_inference_config(self, class_type: Type[T]) -> T:
        return instance_from_dict(class_type, self._config_dict[self._id]["Inference"])
    
    def parse_agent_type(self) -> str:
        return self._config_dict[self._id]["Agent"]["type"]
        
    def parse_agent_config(self, agent_config_class: Type[T]) -> T:
        return instance_from_dict(agent_config_class, self._config_dict[self._id]["Agent"])
    
    def parse_env_config(self, class_type: Type[T]) -> T:
        return instance_from_dict(class_type, self._config_dict[self._id]["Env"])
    
    @staticmethod
    def from_yaml(file_path: str) -> "ConfigParser":
        config_dict = load_yaml(file_path)
        return ConfigParser(config_dict)