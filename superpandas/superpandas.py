import pdb
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import pickle
import json

from .prompts import get_architecture_prompt, get_table_description_prompt
from .llms import TGIClient, OpenAIClient

@dataclass
class ForeignKey:
    """
    Creates a ForeignKey object which adds a foreign key relationship between two columns of a Pandas DataFrame.
    """
    src_sdf: str
    src_column: str
    tgt_sdf: str
    tgt_column: str
     
@dataclass
class SuperDataFrame:
    """
    Creates a SuperDataFrame object which is a wrapper around a Pandas DataFrame. It adds following metadata to the DataFrame:
    - name: name of the table
    - desc: description of the table
    Addtionally it adds dataframe column names and dtypes.
    """
    df: pd.DataFrame
    name: Optional[str] = None
    desc: Optional[str] = None
    
    def __post_init__(self):
        self.columns = list(self.df.columns)
        self.dtypes = {k:str(v) for k,v in self.df.dtypes.items()}
        
    def __eq__(self, __value: object) -> bool:
        return self.df.equals(__value.table)
    
    def equals(self, __value: object) -> bool:
        return self.df.equals(__value.df)
    
    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame
        """
        arch = self.dtypes
        return arch
    
    def to_disk(self, path: Path):
        """
        Saves the SuperDataFrame to disk.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_disk(cls, path: Path):
        """
        Loads the SuperDataFrame from disk.
        """
        with open(path, 'rb') as f:
            cls = pickle.load(f)
            return cls
        
    def set_description(self, text: str, force=False):
        """
        Sets the description of the SuperDataFrame if not already set. If force is True, it replaces the existing description.
        """
        if self.desc is not None and not force:
            raise ValueError("SuperDataFrame already has a description. Use `force=True` to replace it.")
        self.desc = text

class SuperDataFrameV2(pd.DataFrame):
    """
    Creates a SuperDataFrame object which is a wrapper around a Pandas DataFrame. It adds following optional metadata to the DataFrame:
    - name: name of the table
    - desc: description of the table
    - foreign_keys: list of foreign keys
    Addtionally it adds dataframe column names and dtypes.
    """
    
    _metadata = ["name", "desc", "foreign_keys"]
    
    def __init__(self,
                 *args, **kwargs,
                 ) -> None:
        self.name = kwargs.pop('name',None)
        self.desc = kwargs.pop('desc',None)
        self.foreign_keys = kwargs.get('foreign_keys',None)        
        super().__init__(*args, **kwargs,)

    @property
    def _constructor(self):
        return SuperDataFrame
    
    # def __repr__(self): #TODO: Fix this
    #     # df_repr = super().__repr__()
    #     pdb.set_trace()
    #     return {self.name: repr(super())}
    
    # def __str__(self):
    #     # str_ = f"SuperDataFrame:\nname: {self.name}\ndescription: {self.desc}\n"+super().__str__()
    #     return super().__str__()

    def __setattr__(self, attr, val): # https://github.com/geopandas/geopandas/blob/514f975298b940fca1a39917ff35aa12b149a1e7/geopandas/geodataframe.py#L198C1-L203C43
        # have to special case b/c pandas tries to use as column...
        if attr in ["name","desc","foreign_keys"]:
            object.__setattr__(self, attr, val)
        else:
            super().__setattr__(attr, val)
            
    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame
        """
        arch = self.dtypes
        if self.foreign_keys is not None:
            arch['foreign_keys'] = {}
            for fk in self.foreign_keys:
                arch['foreign_keys'][fk.src_column] = (fk.tgt_sdf,fk.tgt_column)
        
        return arch
    
    def to_disk(self, path: Path):
        """
        Saves the SuperDataFrame to disk.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_disk(cls, path: Path):
        """
        Loads the SuperDataFrame from disk.
        """
        with open(path, 'rb') as f:
            cls = pickle.load(f)
            return cls
        
    def set_description(self, text: str, force=False):
        """
        Sets the description of the SuperDataFrame if not already set. If force is True, it replaces the existing description.
        """
        if self.desc is not None and not force:
            raise ValueError("SuperDataFrame already has a description. Use `force=True` to replace it.")
        self.desc = text
        
class PandaPack:
    """
    Creates a PandaPack object which is a collection of SuperDataFrames and optional foreign keys to connect them.
    """
    
    def __init__(self, 
                 sdf: Optional[Union[SuperDataFrame,pd.DataFrame,List[SuperDataFrame],List[pd.DataFrame]]]=None, 
                 summary: Optional[str] = None,
                 foreign_keys: Optional[Union[ForeignKey,List[ForeignKey]]] = None
                 ):
                   
        if isinstance(sdf, list):
            if isinstance(sdf[0],SuperDataFrame):
                self.sdf = sdf
            else: # list of pd.DataFrames
                self.sdf = [SuperDataFrame(df=table) for table in sdf]
        elif isinstance(sdf, SuperDataFrame):
            self.sdf = [sdf]
        elif isinstance(sdf, pd.DataFrame):
            self.sdf = [SuperDataFrame(df=sdf)]
        elif sdf==None:
            self.sdf = {}
        else:
            raise ValueError(f"tables must be a SuperDataFrame, pd.DataFrame, or a list of SuperDataFrames/pd.DataFrames. Received {type(sdf)}")
        
        self.num_tables_wo_name = len([table for table in self.sdf if table.name==None])
        for i in range(self.num_tables_wo_name):
            if self.sdf[i].name is None:
                self.sdf[i].name = f"table_{i}"
        self.sdfs = {table.name: table for table in self.sdf}
        self.summary = summary
        self.foreign_keys= [] if foreign_keys is None else [foreign_keys] if isinstance(foreign_keys, ForeignKey) else foreign_keys
        
        self.verify()

    def __repr__(self):
        return str(self.get_architecture())
    
    def __str__(self):
        return json.dumps(self.get_architecture(),indent=2)
    
    def __eq__(self, __value: object) -> bool:
        for table_name in self.sdf:
            if not self.sdf[table_name].equals(__value.tables[table_name]):
                return False
        return True
    
    def add_sdf(self, df: Union[SuperDataFrame,pd.DataFrame]):
        """
        Adds a pd.DataFrame/SuperDataFrame to PandaPack.
        """
        if isinstance(df, pd.DataFrame):
            table_name = f"table_{self.num_tables_wo_name}"
            df = SuperDataFrame(df=df, name=table_name)
            self.num_tables_wo_name+=1
        assert self.sdf.get(df.name) is None, f"Table {df.name} already exists."
        self.sdfs[df.name] = df
        
    def pop_sdf(self, table_name: str):
        """
        Removes a table with the given name.
        """
        return self.sdfs.pop(table_name,None)
    
    def get_sdf(self, sdf_name: str):
        """
        Returns a table with the given name.
        """
        return self.sdfs.get(sdf_name,None)
    
    def update_sdf(self, sdf_name: str, df: Union[SuperDataFrame,pd.DataFrame]):
        """
        Updates a table with the given name.
        """
        assert self.get_sdf(sdf_name) is not None, f"Table {sdf_name} does not exist."
        self.sdfs[sdf_name] = df
    
    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame in the form of a dictionary, that can be used in a query to the LLM.
        """
        arch = {}
        for name,sdf in self.sdfs.items():
            arch[name] = sdf.get_architecture()
            
        if len(self.foreign_keys)>0:
            arch['foreign_keys'] = []
            for fk in self.foreign_keys:
                arch['foreign_keys'].append(((fk.src_sdf,fk.src_column),(fk.tgt_sdf,fk.tgt_column)))
        return arch
    
    def get_sdf_names(self):
        """
        Returns a list of table names.
        """
        return list(self.sdfs.keys())
    
    def verify(self):
        pass
    
    def add_foreign_key(self,
                        fk: Optional[ForeignKey]=None, 
                        src_sdf: Optional[str]=None, 
                        src_column: Optional[str]=None,
                        tgt_sdf: Optional[str]=None,
                        tgt_column: Optional[str]=None):
        """
        Adds a foreign key between two tables.
        """
        assert fk is not None or (src_sdf is not None and src_column is not None and tgt_sdf is not None and tgt_column is not None), "Either `fk` or `src_sdf`, `src_column`, `tgt_sdf`, and `tgt_column` must be provided."
        
        if fk is None:
            assert self.get_sdf(src_sdf) is not None, f"Table {src_sdf} does not exist."
            assert self.get_sdf(tgt_sdf) is not None, f"Table {tgt_sdf} does not exist."
            assert src_column in self.get_sdf(src_sdf).columns, f"Column {src_column} does not exist in table {src_sdf}."
            assert tgt_column in self.get_sdf(tgt_sdf).columns, f"Column {tgt_column} does not exist in table {tgt_sdf}."
        
            fk = ForeignKey(src_sdf, src_column, tgt_sdf, tgt_column)
            
        self.foreign_keys.append(fk)
        
    def set_summary(self, summary: str, force=False):
        """
        Sets the summary of the PandaPack if not already set. If `force` is True, it replaces the existing summary.
        """
        if self.summary is not None and not force:
            raise ValueError("SuperDataFrame already has a summary. Use `force=True` to replace it.")
        self.summary = summary
        
    def get_summary(self):
        """
        Returns a summary of the SuperDataFrame.
        """
        return self.summary
    
    def to_disk(self, path: Path):
        """
        Saves the SuperDataFrame to disk.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def from_disk(cls, path: Path):
        """
        Loads the SuperDataFrame from disk.
        """
        with open(path, 'rb') as f:
            cls = pickle.load(f)
            return cls

@dataclass
class SuperPandasConfig:
    """
    Creates a Config object which is a dictionary of configuration parameters for a SuperPandas.
    """
    api_key: Optional[str]=None
    llm_type: Optional[str]=None
    llm_id: Optional[str]=None
    openai_max_retries: Optional[int]=None
    openai_base_url: Optional[str]=None
    timeout: Optional[float]=None

@dataclass
class PromptTemplate:
    """
    Creates a PromptTemplate object which is a dictionary of prompts for querying the LLM.
    """
    prompt: str
    
    def __post_init__(self):
        assert 'arch' in self.prompt, r"Prompt must contain the string '{arch}' which will be replaced by the architecture of the SuperDataFrame."
        self.prompt = self.prompt.strip()
   
class SuperPandas:
    
    def __init__(self,
                 pdp: Optional[PandaPack]=None,
                 config: Optional[SuperPandasConfig] = None,
                 ):
        
        self.pdp = pdp
        self.config = config
        
        if self.config.llm_type=='openai':
            self.llm = OpenAIClient(api_key=self.config.api_key,
                                    base_url=self.config.openai_base_url,
                                    max_retries=self.config.openai_max_retries,
                                    model="gpt-3.5-turbo" if self.config.llm_id is None else self.config.llm_id)
            
        elif self.config.llm_type=='tgi':
            self.llm = TGIClient(api_key=self.config.api_key,
                                 model=self.config.llm_id,
                                 timeout=self.config.timeout)
            
        elif self.config.llm_type=='vllm':
            raise NotImplementedError
        
        else:
            ValueError(f"llm_type must be one of 'openai', 'tgi', or 'vllm'. Received {self.config.llm_type}")
        
    def set_pdp(self, pdp: PandaPack, force=False):
        """
        Sets the PandaPack for the SuperPandas instance. If `force` is True, it replaces the existing PandaPack.
        """
        
        if self.pdp is not None and not force:
            raise ValueError("This SuperPandas instance already has a PandaPack. Use `force=True` to replace it.")
        self.pdp = pdp
    
    def __str__(self):
        return self.pdp.__str__()
    
    def __repr__(self):
        return self.pdp.__repr__()
    
    def get_pdp_summary_from_llm(self):
        """
        Queries the LLM for a summary of the PandaPack.
        """
        query = get_architecture_prompt
        
        return self.query(query)
    
    def get_sdf_description_from_llm(self, table_name: str):
        """
        Queries the LLM for a description of the given SDF.
        """
        arch = {table_name:self.pdp.get_sdf(table_name).get_architecture()}
        query = get_table_description_prompt.format(arch=arch)
        
        return self.query(query,add_arch=False)
    
    def query(self, prompt: str, add_arch=True):
        """
        Queries the LLM with given prompt
        """
        arch = self.pdp.get_architecture()
        query = prompt.format(arch=arch) if add_arch else prompt
        
        return self.llm(query)