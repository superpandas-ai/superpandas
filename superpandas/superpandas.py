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
    - foreign_keys: list of foreign keys
    Addtionally it adds dataframe column names and dtypes.
    """
    table: pd.DataFrame
    name: Optional[str] = None
    desc: Optional[str] = None
    foreign_keys: Optional[List[ForeignKey]] = None
    
    def __post_init__(self):
        self.columns = list(self.table.columns)
        self.dtypes = {k:str(v) for k,v in self.table.dtypes.items()}
        
    def __eq__(self, __value: object) -> bool:
        return self.table.equals(__value.table)
    
    def equals(self, __value: object) -> bool:
        return self.table.equals(__value.table)
    
    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame
        """
        # arch = {}
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
    
class PandaPack:
    """
    Creates a PandaPack object which is a collection of SuperDataFrames.
    """
    
    def __init__(self, 
                 tables: Optional[Union[SuperDataFrame,pd.DataFrame,List[SuperDataFrame],List[pd.DataFrame]]]=None, 
                 summary: Optional[str] = None):
                   
        if isinstance(tables, list):
            if isinstance(tables[0],SuperDataFrame):
                self.tables = tables
            else: # list of pd.DataFrames
                self.tables = [SuperDataFrame(table=table) for table in tables]
        elif isinstance(tables, SuperDataFrame):
            self.tables = [tables]
        elif isinstance(tables, pd.DataFrame):
            self.tables = [SuperDataFrame(table=tables)]
        elif tables==None:
            self.tables = []
        else:
            raise ValueError(f"tables must be a SuperDataFrame, pd.DataFrame, or a list of SuperDataFrames/pd.DataFrames. Received {type(tables)}")
        
        self.num_tables_wo_name = len([table for table in self.tables if table.name==None])
        for i in range(self.num_tables_wo_name):
            if self.tables[i].name is None:
                self.tables[i].name = f"table_{i}"
        self.tables = {table.name: table for table in self.tables}
        self.summary = summary
        self.verify()

    def __repr__(self):
        return str(self.get_architecture())
    
    def __str__(self):
        return json.dumps(self.get_architecture(),indent=2)
    
    def __eq__(self, __value: object) -> bool:
        for table_name in self.tables:
            if not self.tables[table_name].equals(__value.tables[table_name]):
                return False
        return True
    
    def add_table(self, table: Union[SuperDataFrame,pd.DataFrame]):
        """
        Adds a table to the SuperDataFrame."""
        if isinstance(table, pd.DataFrame):
            table_name = f"table_{self.num_tables_wo_name}"
            table = SuperDataFrame(table=table, name=table_name)
            self.num_tables_wo_name+=1
        self.tables[table.name] = table
        
    def pop_table(self, table_name: str):
        """
        Removes a table with the given name.
        """
        return self.tables.pop(table_name,None)
    
    def get_table(self, table_name: str):
        """
        Returns a table with the given name.
        """
        return self.tables.get(table_name,None)
    
    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame in the form of a dictionary, that can be used in a query to the LLM.
        """
        arch = {}
        for name,table in self.tables.items():
            arch[name] = table.get_architecture()
        return arch
    
    def get_table_names(self):
        """
        Returns a list of table names.
        """
        return list(self.tables.keys())
    
    def verify(self):
        pass
    
    def add_foreign_key(self, src_sdf: str, src_column: str, tgt_sdf: str, tgt_column: str):
        """
        Adds a foreign key between two tables.
        """
        assert self.get_table(src_sdf) is not None, f"Table {src_sdf} does not exist."
        assert self.get_table(tgt_sdf) is not None, f"Table {tgt_sdf} does not exist."
        assert src_column in self.get_table(src_sdf).columns, f"Column {src_column} does not exist in table {src_sdf}."
        assert tgt_column in self.get_table(tgt_sdf).columns, f"Column {tgt_column} does not exist in table {tgt_sdf}."
        
        src_table = self.get_table(src_sdf)
        
        fk = ForeignKey(src_sdf, src_column, tgt_sdf, tgt_column)
        if src_table.foreign_keys is None:
            src_table.foreign_keys = [fk]
        else:
            src_table.foreign_keys.append(fk)
    
    def add_summary(self, summary: str):
        if self.summary is not None:
            raise ValueError("SuperDataFrame already has a summary. Use SuperDataFrame.set_summary() to replace it.")
        self.summary = summary
        
    def set_summary(self, summary: str):
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
    Creates a Config object which is a dictionary of configuration parameters for a SuperDataFrame.
    """
    api_key: Optional[str]=None
    llm_type: Optional[str]=None
    llm_id: Optional[str]=None
    openai_max_retries: Optional[int]=None
    openai_base_url: Optional[str]=None
    timeout: Optional[float]=None
    
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
        
    def add_pdp(self, pdp: PandaPack):
        if self.pdp is not None:
            raise ValueError("This SuperPandas instance already has a PandaPack. Use SuperPandas.set_pdp() to replace it.")
        self.pdp = pdp
        
    def set_pdp(self, pdp: PandaPack):
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
    
    def get_table_description_from_llm(self, table_name: str):
        """
        Queries the LLM for a description of the given table.
        """
        
        arch = {table_name:self.pdp.get_table(table_name).get_architecture()}
        query = get_table_description_prompt.format(arch=arch)
        
        return self.query(query,add_arch=False)
    
    def query(self, prompt: str, add_arch=True):
        """
        Queries the LLM with given prompt
        """
        arch = self.pdp.get_architecture()
        query = prompt.format(arch=arch) if add_arch else prompt
        
        return self.llm(query)