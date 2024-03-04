# References: https://github.com/aspuru-guzik-group/curiosity/blob/main/selfies_helper.py

def __selfies_to_smiles_derive(selfies,smiles,N_restrict=True):    
    # Elements of start_alphabet, again, stand for integers (see comments in _smiles_to_selfies function for more details)
    start_alphabet=['[#N]','[epsilon]','[Ring1]','[Ring2]','[Branch1_1]','[Branch1_2]','[Branch1_3]','[Branch2_1]','[Branch2_2]','[Branch2_3]','[F]','[O]','[=O]','[N]','[=N]','[#N]','[C]','[=C]','[#C]','[S]','[=S]'];
            
    tmp_ds=selfies.replace('X','Z!') # X will be used as states of the derivation
    
    next_X=smiles.find('X');
    while next_X>=0:
        before_smiles=smiles[0:next_X] # smiles before the non-terminal
        if smiles[next_X+1]=='9':
            state=int(smiles[next_X+1:next_X+5])  # states after branches are called X999...
            after_smiles=smiles[next_X+6:] # smiles after the non-terminal            
        else:            
            state=int(smiles[next_X+1]) # the state is given by the nonterminal symbol X_n, where n=state        
            after_smiles=smiles[next_X+2:] # smiles after the non-terminal
            
        [current_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds) # the current selfies symbol gives the rule vector, and the current state indentifies the one specific, current rule.
        
        # The semantic informations of this set of rules could be significantly extended and more details could be added. Here, we have semantic rules for the most important molecules in organic chemistry, Carbon, Oxygen, Nitrogen, Flour.
        # Other elements get a generic (very weak) restrictions

        if state==0:
            if current_symbol=='[epsilon]':
                new_smiles_symbol='X0'
            elif current_symbol.find('Ring')>=0 or current_symbol.find('Branch')>=0:
                new_smiles_symbol='X0'
                [_,tmp_ds]=_get_next_selfies_symbol(tmp_ds)  # ignore next symbol  
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]X1'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]X1'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]X1'
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X2'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[O]X2'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X3'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[N]X3'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[N]X3'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X4'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[C]X4'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[C]X4'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X6'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[S]X6'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles

        if state==1:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''
            elif current_symbol.find('Branch')>=0:   
                new_smiles_symbol='X1'
                [_,tmp_ds]=_get_next_selfies_symbol(tmp_ds)  # ignore next symbol               
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'
                
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')
            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]                
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'
                    
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[S]X5'
            else:
                new_smiles_symbol=current_symbol+'X6'                
            smiles=before_smiles+new_smiles_symbol+after_smiles

        if state==2:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''              
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]                     
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'
                    
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X1'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X1'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X1'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]                      
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'
                    
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X1'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X1'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X1'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[Branch1_1]' or current_symbol=='[Branch1_2]' or current_symbol=='[Branch1_3]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'

            elif current_symbol=='[Branch2_1]' or current_symbol=='[Branch2_2]' or current_symbol=='[Branch2_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'
                

            elif current_symbol=='[Branch3_1]' or current_symbol=='[Branch3_2]' or current_symbol=='[Branch3_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol

                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'              
                
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles


        if state==3:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''             
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]                  
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'
                    
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X2'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X2'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X2'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]                    
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'
                    
                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X2'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X2'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X2'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[Branch1_1]' or current_symbol=='[Branch1_2]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                
            elif current_symbol=='[Branch1_3]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'             
                
            elif current_symbol=='[Branch2_1]' or current_symbol=='[Branch2_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                
                
            elif current_symbol=='[Branch2_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'
                
                

            elif current_symbol=='[Branch3_1]' or current_symbol=='[Branch3_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                
                
            elif current_symbol=='[Branch3_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'                 
                
                
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[#N]'
                else:
                    new_smiles_symbol='[#N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[#C]X1'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'

            smiles=before_smiles+new_smiles_symbol+after_smiles


        if state==4:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''            
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'                

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X3'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X3'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X3'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                new_smiles_symbol='X4'
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X3'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X3'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X3'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[Branch1_1]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                
            elif current_symbol=='[Branch1_2]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'              
                
            elif current_symbol=='[Branch1_3]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'
                
            elif current_symbol=='[Branch2_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                

            elif current_symbol=='[Branch2_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'            
                
                
            elif current_symbol=='[Branch2_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)   
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'
                
            elif current_symbol=='[Branch3_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                

            elif current_symbol=='[Branch3_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'         
                
                
            elif current_symbol=='[Branch3_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X1'         



            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[#N]'
                else:
                    new_smiles_symbol='[#N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[#C]X1'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles



        if state==5:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''            
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'                

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X4'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X4'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X4'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                new_smiles_symbol='X5'
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X4'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X4'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X4'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[Branch1_1]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'
                
            elif current_symbol=='[Branch1_2]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1

                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'              

            elif current_symbol=='[Branch1_3]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1

                branch_str=''
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                
            elif current_symbol=='[Branch2_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'
                

            elif current_symbol=='[Branch2_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'            
                
                
            elif current_symbol=='[Branch2_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)   
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'
                                
            elif current_symbol=='[Branch3_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'
                
            elif current_symbol=='[Branch3_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'         
                
            elif current_symbol=='[Branch3_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X2'     

                
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[#N]'
                else:
                    new_smiles_symbol='[#N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[#C]X1'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles




        if state==6:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''            
            elif current_symbol.find('Ring1]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    ring_num=str(start_alphabet.index(next_symbol)+2)
                else:
                    ring_num='5'                

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X5'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X5'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X5'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol.find('Ring2]')>=0:
                pre_symbol=''
                if current_symbol[1:5]=='Expl': # Explicit Bond Information
                    pre_symbol=current_symbol[5]
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                new_smiles_symbol='X6'
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    ring_num_1=(start_alphabet.index(next_symbol1)+1)*20
                    ring_num_2=start_alphabet.index(next_symbol2)                
                    ring_num=str(ring_num_1+ring_num_2)
                else:
                    ring_num='5'

                if len(ring_num)==1:
                    new_smiles_symbol=pre_symbol+'%00'+ring_num+'X5'
                elif len(ring_num)==2:
                    new_smiles_symbol=pre_symbol+'%0'+ring_num+'X5'
                elif len(ring_num)==3:
                    new_smiles_symbol=pre_symbol+'%'+ring_num+'X5'
                else:
                    raise ValueError('__selfies_to_smiles_derive: Problem with deriving very long ring.')

            elif current_symbol=='[Branch1_1]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'
                
            elif current_symbol=='[Branch1_2]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1

                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X5'              

            elif current_symbol=='[Branch1_3]':
                [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if next_symbol in start_alphabet:
                    branch_num=start_alphabet.index(next_symbol)+1
                else:
                    branch_num=1

                branch_str=''
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'
                
            elif current_symbol=='[Branch2_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'
                

            elif current_symbol=='[Branch2_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X5'            
                
                
            elif current_symbol=='[Branch2_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)   
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*20
                    branch_num2=start_alphabet.index(next_symbol2)
                    branch_num=branch_num1+branch_num2
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'
                                
            elif current_symbol=='[Branch3_1]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9992',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X4'
                
            elif current_symbol=='[Branch3_2]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9991',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X5'         
                
            elif current_symbol=='[Branch3_3]':
                [next_symbol1,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol2,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                [next_symbol3,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                if (next_symbol1 in start_alphabet) and (next_symbol2 in start_alphabet) and (next_symbol3 in start_alphabet):
                    branch_num1=(start_alphabet.index(next_symbol1)+1)*400
                    branch_num2=(start_alphabet.index(next_symbol2))*20
                    branch_num3=start_alphabet.index(next_symbol3)
                    branch_num=branch_num1+branch_num2+branch_num3
                else:
                    branch_num=1
                
                branch_str=''                
                for bii in range(branch_num):
                    [next_symbol,tmp_ds]=_get_next_selfies_symbol(tmp_ds)
                    branch_str+=next_symbol
                    
                branch_smiles,_=__selfies_to_smiles_derive(branch_str,'X9993',N_restrict)
                new_smiles_symbol=''
                if len(branch_smiles)>0:
                    new_smiles_symbol='('+branch_smiles+')X3'     

                
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X6'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X6'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[#N]'
                else:
                    new_smiles_symbol='[#N]X6'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[#C]X1'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles



        if state==9991: # states 5-7 occure after branches are derived, because a branch or a ring directly after a branch is syntactically illegal.
                     # state  5 corresponds to state 1, state 6 corresponds to state 2, and state 7 corresponds to state 3, without branches & rings
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''
            elif current_symbol.find('Ring')>=0 or current_symbol.find('Branch')>=0: 
                new_smiles_symbol='X9991'
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X4'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X4'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X4'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[S]X5'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles



        if state==9992:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''          
            elif current_symbol.find('Ring')>=0 or current_symbol.find('Branch')>=0: 
                new_smiles_symbol='X9992' 
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X4'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X4'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X4'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'
            smiles=before_smiles+new_smiles_symbol+after_smiles




        if state==9993:
            if current_symbol=='[epsilon]':
                new_smiles_symbol=''        
            elif current_symbol.find('Ring')>=0 or current_symbol.find('Branch')>=0: 
                new_smiles_symbol='X9993'     
            elif current_symbol=='[F]':
                new_smiles_symbol='[F]'
            elif current_symbol=='[Cl]':
                new_smiles_symbol='[Cl]'
            elif current_symbol=='[Br]':
                new_smiles_symbol='[Br]'                  
            elif current_symbol=='[O]':
                new_smiles_symbol='[O]X1'
            elif current_symbol=='[=O]':
                new_smiles_symbol='[=O]'
            elif current_symbol=='[N]':
                if N_restrict:
                    new_smiles_symbol='[N]X2'
                else:
                    new_smiles_symbol='[N]X4'
            elif current_symbol=='[=N]':
                if N_restrict:
                    new_smiles_symbol='[=N]X1'
                else:
                    new_smiles_symbol='[=N]X4'
            elif current_symbol=='[#N]':
                if N_restrict:
                    new_smiles_symbol='[#N]'
                else:
                    new_smiles_symbol='[#N]X4'
            elif current_symbol=='[C]':
                new_smiles_symbol='[C]X3'
            elif current_symbol=='[=C]':
                new_smiles_symbol='[=C]X2'
            elif current_symbol=='[#C]':
                new_smiles_symbol='[#C]X1'
            elif current_symbol=='[S]':
                new_smiles_symbol='[S]X5'
            elif current_symbol=='[=S]':
                new_smiles_symbol='[=S]X4'
            else:
                new_smiles_symbol=current_symbol+'X6'

            smiles=before_smiles+new_smiles_symbol+after_smiles
    
        if len(tmp_ds)<=2: # if all selfies symbols are derived, the final non-terminals are removed
            while True:
                non_terminal=smiles.find('X')
                if non_terminal>=0:
                    if smiles[non_terminal+1]=='9':
                        smiles=smiles[0:non_terminal]+smiles[non_terminal+5:]
                    else:
                        smiles=smiles[0:non_terminal]+smiles[non_terminal+2:]
                else:
                    break;

        next_X=smiles.find('X')
        
    return smiles.replace('Z!','X'), tmp_ds



def _get_next_selfies_symbol(tmp_ds): # get the next selfies symbol
    next_symbol=''
    tmp_ds_new=tmp_ds
    if len(tmp_ds)<=2:
        return [next_symbol, tmp_ds_new]
    
    if tmp_ds[0]!='[':
        raise ValueError('_get_next_selfies_symbol: Decoding Problem 1: '+tmp_ds)
    
    end_of_symbol=tmp_ds.find(']')
    
    if end_of_symbol==-1:
        raise ValueError('_get_next_selfies_symbol: Decoding Problem 2: '+tmp_ds)
    else:
        next_symbol=tmp_ds[0:end_of_symbol+1]
        tmp_ds_new=tmp_ds_new[end_of_symbol+1:]
        
    return [next_symbol, tmp_ds_new]





def is_finished(selfies):
    selfies += '[Q]'
    _,tmp_ds = __selfies_to_smiles_derive(selfies, 'X0')
    
    if len(tmp_ds) > 0:
        return True
    else:
        return False


#selfies = '[P][STOP]'
#print(is_finished(selfies))
