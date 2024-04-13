import pandas as pd
from pathlib import Path
import sys
import random
sys.path.append(str(Path(__file__).parent.parent/"superpandas"))
from superpandas import SuperDataFrameV2, SuperDataFrame, PandaPack, SuperPandas, SuperPandasConfig, ForeignKey

random_integers = lambda : [random.randint(1, 100) for _ in range(3)]

# Tests for the PandaPack class
def sample_df():
    data = {'A': random_integers(), 'B': random_integers()}
    return pd.DataFrame(data)

def test_super_dataframe_init_with_single_super_table():
    sdf = SuperDataFrame(name='Table1', df=sample_df())
    pdp = PandaPack(sdf=sdf)
    assert len(pdp.sdfs) == 1
    assert 'Table1' in pdp.sdfs

def test_super_dataframe_init_with_single_pd_dataframe():
    df = sample_df()
    pdp = PandaPack(sdf=df)
    assert len(pdp.sdfs) == 1
    assert 'table_0' in pdp.sdfs

def test_super_dataframe_init_with_list_of_tables():
    sdf1 = SuperDataFrame(name='Table1', df=sample_df())
    sdf2 = SuperDataFrame(name='Table2', df=sample_df())
    pdp = PandaPack(sdf=[sdf1, sdf2])
    assert len(pdp.sdfs) == 2
    assert 'Table1' in pdp.sdfs
    assert 'Table2' in pdp.sdfs

def test_add_dataframe_without_name():
    pdp = PandaPack(sdf=sample_df())
    new_sdf = SuperDataFrame(name='NewTable', df=sample_df())
    pdp.add_sdf(new_sdf)
    assert len(pdp.sdfs) == 2
    assert 'table_0' in pdp.sdfs
    assert 'NewTable' in pdp.sdfs
    
def test_add_dataframe_with_name():
    pdp = PandaPack(sdf=sample_df())
    df=sample_df()
    pdp.add_sdf(df)
    assert len(pdp.sdfs) == 2
    assert 'table_0' in pdp.sdfs
    assert 'table_1' in pdp.sdfs

def test_get_table():
    sdf = SuperDataFrame(name='table_0', df=sample_df())
    pdp = PandaPack(sdf=sdf)
    retrieved_sdf = pdp.get_sdf('table_0')
    assert retrieved_sdf.equals(sdf)

def test_get_table_names():
    table1 = SuperDataFrame(name='Table1', df=sample_df())
    table2 = SuperDataFrame(name='Table2', df=sample_df())
    sdf = PandaPack(sdf=[table1, table2])
    names = sdf.get_sdf_names()
    assert 'Table1' in names
    assert 'Table2' in names

def test_add_foreign_key():
    sdf1 = SuperDataFrame(name='Table1', df=sample_df())
    sdf2 = SuperDataFrame(name='Table2', df=sample_df())
    pdp = PandaPack(sdf=[sdf1, sdf2])
    pdp.add_foreign_key(src_sdf='Table1', src_column='A', tgt_sdf='Table2', tgt_column='B')
    assert len(pdp.foreign_keys) == 1
    fk = ForeignKey(src_sdf='Table1', src_column='B', tgt_sdf='Table2', tgt_column='A')
    pdp.add_foreign_key(fk)
    assert len(pdp.foreign_keys) == 2

def test_to_and_from_disk(tmp_path):
    sdf = SuperDataFrame(name='table_0', df=sample_df())
    pdp = PandaPack(sdf=sdf)
    file_path = tmp_path / "test_sdf.pkl"
    pdp.to_disk(file_path)
    
    loaded_pdp = PandaPack.from_disk(file_path)
    
    assert len(loaded_pdp.sdf) == 1
    assert 'table_0' in loaded_pdp.sdfs
    assert loaded_pdp.get_sdf('table_0').equals(sdf)
    
def test_openai_client():
    config = SuperPandasConfig(llm_type='openai')
    sdf1 = SuperDataFrame(name='Table1', df=sample_df())
    sdf2 = SuperDataFrame(name='Table2', df=sample_df())
    pdp = PandaPack(sdf=[sdf1, sdf2])
    pdp.add_foreign_key(src_sdf='Table1', src_column='A', tgt_sdf='Table2', tgt_column='B')
    spd = SuperPandas(pdp=pdp, config=config)
    output = spd.get_pdp_summary_from_llm()
    response = output.choices[0].message.content

    assert len(response) > 20 # The response should be a non-empty string
    assert 'Table1' in response  # The response should contain the name of the table
    assert 'Table2' in response  # The response should contain the name of the table
    assert 'A' in response  # The response should contain the name of the column
    assert 'B' in response  # The response should contain the name of the column
    assert 'int64' in response  # The response should contain the dtype of the column

def test_tgi_client():
    config = SuperPandasConfig(llm_type='tgi')
    sdf1 = SuperDataFrame(name='Table1', df=sample_df())
    sdf2 = SuperDataFrame(name='Table2', df=sample_df())
    pdp = PandaPack(sdf=[sdf1, sdf2])
    pdp.add_foreign_key(src_sdf='Table1', src_column='A', tgt_sdf='Table2', tgt_column='B')
    spd = SuperPandas(pdp=pdp, config=config)
    response = spd.get_pdp_summary_from_llm()
    
    assert len(response) > 20 # The response should be a non-empty string
    assert 'Table1' in response  # The response should contain the name of the table
    assert 'Table2' in response  # The response should contain the name of the table
    assert 'A' in response  # The response should contain the name of the column
    assert 'B' in response  # The response should contain the name of the column
    # assert 'int64' in response  # The response should contain the dtype of the column
    
def test_tgi_client_table_description():
    config = SuperPandasConfig(llm_type='tgi')
    sdf = SuperDataFrame(name='Table1', df=sample_df())
    pdp = PandaPack(sdf=sdf)
    spd = SuperPandas(pdp=pdp, config=config)
    response = spd.get_sdf_description_from_llm('Table1')
    
    assert len(response) > 20 # The response should be a non-empty string
    assert 'Table1' in response  # The response should contain the name of the table
    assert 'A' in response  # The response should contain the name of the column
    assert 'B' in response  # The response should contain the name of the column
    assert '64' in response  # The response should contain the dtype of the column