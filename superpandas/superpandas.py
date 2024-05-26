import pdb
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Union, Optional, Any
from pathlib import Path
import pickle
import json

from .prompts import get_architecture_prompt, get_table_description_prompt
from .llm_clients import TGIClient, OpenAIClient


def json_dumper(obj):
    try:
        return obj.to_json()
    except:
        return str(obj)


@dataclass
class ForeignKey:
    """
    Creates a ForeignKey object which adds a foreign key relationship between two columns of a Pandas DataFrame.
    """
    src_sdf: str
    src_column: str
    tgt_sdf: str
    tgt_column: str

    def __eq__(self, value: object) -> bool:
        return (self.src_sdf == value.src_sdf and
                self.src_column == value.src_column and
                self.tgt_sdf == value.tgt_sdf and
                self.tgt_column == value.tgt_column)


@dataclass
class SuperDataFrameV1:
    """
    Creates a SuperDataFrame object which is a wrapper around a Pandas DataFrame. It adds following metadata to the DataFrame:
    - name: name of the table
    - descrption: description of the table
    Addtionally it adds dataframe column names and dtypes.
    """
    df: pd.DataFrame
    name: Optional[str] = None
    descrption: Optional[str] = None

    def __post_init__(self):
        self.columns = list(self.df.columns)
        self.dtypes = {k: str(v) for k, v in self.df.dtypes.items()}

    def __eq__(self, __value: object) -> bool:
        return self.df.equals(__value.df)

    def equals(self, __value: object) -> bool:
        return self.df.equals(__value.df)

    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame
        """
        arch = {'name': self.name,
                'descrption': self.descrption}
        arch.update(self.dtypes)

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
        if self.descrption is not None and not force:
            raise ValueError(
                "SuperDataFrame already has a description. Use `force=True` to replace it.")
        self.descrption = text


class SuperDataFrame(pd.DataFrame):
    """
    Creates a SuperDataFrame object which is a subclass of Pandas DataFrame. It adds following optional metadata to the DataFrame:
    - name: name of the table
    - descrption: description of the table

    Additionally it introduces following methods:
    - get_architecture: Returns the architecture of SuperDataFrame
    - to_disk: Saves the SuperDataFrame to disk
    - from_disk: Loads the SuperDataFrame from disk
    - set_description: Sets the description of the SuperDataFrame if not already set. If force=True is passed, it replaces the existing description.
    """

    _metadata = ["name", "descrption"]

    name = None
    descrption = None

    def __init__(self,
                 *args, **kwargs,
                 ) -> None:
        self.name = kwargs.pop('name', None)
        self.descrption = kwargs.pop('descrption', None)
        super().__init__(*args, **kwargs,)

    @property
    def _constructor(self):
        return SuperDataFrame

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            if self.name != other.name or self.descrption != other.descrption:
                return False
            if not super().equals(other):  # TODO: Check if this is correct for dataframes
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # def __repr__(self): #TODO: Fix this
    #     # df_repr = super().__repr__()
    #     pdb.set_trace()
    #     return {self.name: repr(super())}

    # def __str__(self): #TODO: complete it

    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame
        """
        arch = {'name': self.name,
                'descrption': self.descrption}
        arch['arch'] = dict(self.dtypes)

        return arch

    def to_disk(self, path: Path):
        """
        Pickles the SuperDataFrame to disk.
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
        if self.descrption != "" and not force:
            raise ValueError(
                "SuperDataFrame already has a description. Use `force=True` to replace it.")
        self.descrption = text


class PandaPack:
    """
    Creates a PandaPack object which is a collection of SuperDataFrames and optional foreign keys to connect them.
    """

    def __init__(self,
                 sdf: Optional[Union[SuperDataFrame, pd.DataFrame,
                                     List[SuperDataFrame], List[pd.DataFrame]]] = None,
                 summary: Optional[str] = None,
                 foreign_keys: Optional[Union[ForeignKey,
                                              List[ForeignKey]]] = None
                 ):

        if isinstance(sdf, list):
            if isinstance(sdf[0], SuperDataFrame):
                self.sdfs = sdf
            else:  # list of pd.DataFrames
                self.sdfs = [SuperDataFrame(table) for table in sdf]
        elif isinstance(sdf, SuperDataFrame):
            self.sdfs = [sdf]
        elif isinstance(sdf, pd.DataFrame):
            self.sdfs = [SuperDataFrame(sdf)]
        elif sdf == None:
            self.sdfs = {}
        else:
            raise ValueError(
                f"tables must be a SuperDataFrame, pd.DataFrame, or a list of SuperDataFrames/pd.DataFrames. Received {type(sdf)}")

        # In case of pd.DataFrames, assign names to them consecutively
        self.num_tables_wo_name = len(
            [table for table in self.sdfs if table.name == None])
        for i in range(self.num_tables_wo_name):
            if self.sdfs[i].name is None:
                self.sdfs[i].name = f"table_{i}"

        self.sdfs = {table.name: table for table in self.sdfs}
        self.summary = summary
        self.foreign_keys = [] if foreign_keys is None else [
            foreign_keys] if isinstance(foreign_keys, ForeignKey) else foreign_keys

        self.verify()  # TODO: Implement this

    def __repr__(self):
        return self.to_json(output_string=True)

    def __str__(self):
        return str(self.to_json())

    def __eq__(self, other: object) -> bool:
        for sdf in self.sdfs:
            if not self.sdfs[sdf] == other.sdfs[sdf]:
                return False
        for fk in self.foreign_keys:
            if fk not in other.foreign_keys:
                return False
            if fk != other.foreign_keys[fk]:  # TODO: Add test
                return False
            # if (fk.src_sdf!=other.foreign_keys[fk].src_sdf or
            #     fk.src_column!=other.foreign_keys[fk].src_column or
            #     fk.tgt_sdf!=other.foreign_keys[fk].tgt_sdf or
            #     fk.tgt_column!=other.foreign_keys[fk].tgt_column):
            #     return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_sdf(self, df: Union[SuperDataFrame, pd.DataFrame]):
        """
        Adds a pd.DataFrame/SuperDataFrame to PandaPack.
        """
        if hasattr(df, 'name'):  # :TODO check why this doesn't work : isinstance(df, SuperDataFrame):
            assert self.sdfs.get(
                df.name) is None, f"DataFrame {df.name} already exists."
            self.sdfs[df.name] = df
        elif isinstance(df, pd.DataFrame):
            table_name = f"table_{self.num_tables_wo_name}"
            df = SuperDataFrame(df, name=table_name)
            self.num_tables_wo_name += 1
            assert self.sdfs.get(
                df.name) is None, f"DataFrame {df.name} already exists."
            self.sdfs[df.name] = df
        else:
            raise ValueError(
                f"df must be a SuperDataFrame or a pd.DataFrame. Received {type(df)}")

    def pop_sdf(self, table_name: str):
        """
        Removes an SDF with the given name.
        """
        return self.sdfs.pop(table_name, None)

    def get_sdf(self, sdf_name: str):
        """
        Returns an SDF with the given name.
        """
        return self.sdfs.get(sdf_name, None)

    def update_sdf(self, sdf_name: str, df: Union[SuperDataFrame, pd.DataFrame]):
        """
        Updates SDF with the given name.
        """
        assert self.get_sdf(
            sdf_name) is not None, f"Table {sdf_name} does not exist."
        if isinstance(df, SuperDataFrame):
            self.sdfs[sdf_name] = df
        else:  # pd.DataFrame
            df = SuperDataFrame(df, name=sdf_name)
            self.sdfs[sdf_name] = df

    def get_architecture(self):
        """
        Returns the architecture of SuperDataFrame in the form of a dictionary, that can be used in a query to the LLM.
        """
        arch = {}
        for name, sdf in self.sdfs.items():
            arch[name] = sdf.get_architecture()

        if len(self.foreign_keys) > 0:
            arch['foreign_keys'] = []
            for fk in self.foreign_keys:
                arch['foreign_keys'].append(
                    ((fk.src_sdf, fk.src_column), (fk.tgt_sdf, fk.tgt_column)))
        return arch

    def get_sdf_names(self):
        """
        Returns a list of table names.
        """
        return list(self.sdfs.keys())

    def to_json(self, output_string=False, exclude_fk=False):
        """
        Returns the architecture of SuperDataFrame in the form of a dictionary.
        """
        arch = self.get_architecture()
        for name in arch.keys():
            if name == 'foreign_keys':
                if exclude_fk:
                    continue
                # else:
                #     arch[name] = [fk for fk in arch[name]]
            else:
                arch[name]['data'] = self.sdfs[name].to_dict()

        if output_string:
            return json.dumps(arch, default=json_dumper, indent=2)
        else:
            return arch

    def verify(self):  # TODO: Implement tests to ensure PDP is consistent
        pass

    def add_foreign_key(self,
                        fk: Optional[ForeignKey] = None,
                        src_sdf: Optional[str] = None,
                        src_column: Optional[str] = None,
                        tgt_sdf: Optional[str] = None,
                        tgt_column: Optional[str] = None):
        """
        Adds a foreign key between two tables.
        """
        assert fk is not None or (
            src_sdf is not None and src_column is not None and tgt_sdf is not None and tgt_column is not None), "Either `fk` or `src_sdf`, `src_column`, `tgt_sdf`, and `tgt_column` must be provided."

        if fk is None:
            assert self.get_sdf(
                src_sdf) is not None, f"Table {src_sdf} does not exist."
            assert self.get_sdf(
                tgt_sdf) is not None, f"Table {tgt_sdf} does not exist."
            assert src_column in self.get_sdf(
                src_sdf).columns, f"Column {src_column} does not exist in table {src_sdf}."
            assert tgt_column in self.get_sdf(
                tgt_sdf).columns, f"Column {tgt_column} does not exist in table {tgt_sdf}."

            fk = ForeignKey(src_sdf, src_column, tgt_sdf, tgt_column)

        if isinstance(fk, list):
            for fk_ in fk:
                self.foreign_keys.append(fk_)
        else:
            self.foreign_keys.append(fk)

    def set_summary(self, summary: str, force=False):
        """
        Sets the summary of the PandaPack if not already set. If `force` is True, it replaces the existing summary.
        """
        if self.summary is not None and not force:
            raise ValueError(
                "SuperDataFrame already has a summary. Use `force=True` to replace it.")
        self.summary = summary

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

    @classmethod
    # TODO: add ingesting df values as well.
    def from_json(cls, payload: json):
        """
        Initializes a PandaPack from a dictionary of architecture.
        """
        fks = []
        sdfs = []
        if isinstance(payload, str):
            payload = json.loads(payload)
        for key, value in payload.items():
            # pdb.set_trace()
            if 'foreign_keys' in key:
                for val in value:
                    ((src_sdf, src_column), (tgt_sdf, tgt_column)) = val
                    fk = ForeignKey(src_sdf, src_column, tgt_sdf, tgt_column)
                    fks.append(fk)
            else:
                name = key if value.get(
                    'name', None) is None else value['name']
                descrption = value.get('descrption', None)
                if value.get('data', None) is not None:
                    df = pd.DataFrame(value['data'])
                    df = df.astype(value['arch'])
                else:
                    if 'arch' not in value:  # TODO: Remove it later as arch key will be present in all architectures
                        dtypes = {k: v for k, v in value.items() if k not in [
                            'name', 'descrption']}
                    else:
                        dtypes = value['arch']
                    columns = list(dtypes.keys())
                    df = pd.DataFrame(columns=columns)
                    df = df.astype(dtypes)
                sdf = SuperDataFrame(df, name=name, descrption=descrption)
                sdfs.append(sdf)
        return cls(sdf=sdfs, foreign_keys=fks)


@dataclass
class SuperPandasConfig:
    """
    Creates a Config object which is a dictionary of configuration parameters for a SuperPandas.
    """
    api_key: Optional[str] = None
    llm_type: Optional[str] = None
    llm_id: Optional[str] = None
    openai_max_retries: Optional[int] = None
    openai_base_url: Optional[str] = None
    timeout: Optional[float] = None


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
                 pdp: Optional[PandaPack] = None,
                 config: Optional[SuperPandasConfig] = None,
                 ):

        self.pdp = pdp
        self.config = config

        if self.config.llm_type == 'openai':
            self.llm = OpenAIClient(api_key=self.config.api_key,
                                    base_url=self.config.openai_base_url,
                                    max_retries=self.config.openai_max_retries,
                                    model="gpt-3.5-turbo" if self.config.llm_id is None else self.config.llm_id)

        elif self.config.llm_type == 'tgi':
            self.llm = TGIClient(api_key=self.config.api_key,
                                 model=self.config.llm_id,
                                 timeout=self.config.timeout)

        elif self.config.llm_type == 'vllm':
            raise NotImplementedError

        else:
            ValueError(
                f"llm_type must be one of 'openai', 'tgi', or 'vllm'. Received {self.config.llm_type}")

    def set_pdp(self, pdp: PandaPack, force=False):
        """
        Sets the PandaPack for the SuperPandas instance. If `force` is True, it replaces the existing PandaPack.
        """

        if self.pdp is not None and not force:
            raise ValueError(
                "This SuperPandas instance already has a PandaPack. Use `force=True` to replace it.")
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
        arch = {table_name: self.pdp.get_sdf(table_name).get_architecture()}
        query = get_table_description_prompt.format(arch=arch)

        return self.query(query, add_arch=False)

    def query(self, prompt: str, add_arch=True):
        """
        Queries the LLM with given prompt
        """
        arch = self.pdp.get_architecture()
        query = prompt.format(arch=arch) if add_arch else prompt

        return self.llm(query)
