import pandas as pd
from pathlib import Path
import sys
import random
sys.path.append(str(Path(__file__).parent.parent/"src"))
from superpandas import SuperDataFrame, PandaPack, SuperPandas, SuperPandasConfig

random_integers = lambda : [random.randint(1, 100) for _ in range(3)]

# Tests for the PandaPack class
def sample_table():
    data = {'A': random_integers(), 'B': random_integers()}
    return pd.DataFrame(data)

def test_super_dataframe_init_with_single_super_table():
    table = SuperDataFrame(name='Table1', table=sample_table())
    sdf = PandaPack(tables=table)
    assert len(sdf.tables) == 1
    assert 'Table1' in sdf.tables

def test_super_dataframe_init_with_single_pd_dataframe():
    table = sample_table()
    sdf = PandaPack(tables=table)
    assert len(sdf.tables) == 1
    assert 'table_0' in sdf.tables

def test_super_dataframe_init_with_list_of_tables():
    table1 = SuperDataFrame(name='Table1', table=sample_table())
    table2 = SuperDataFrame(name='Table2', table=sample_table())
    sdf = PandaPack(tables=[table1, table2])
    assert len(sdf.tables) == 2
    assert 'Table1' in sdf.tables
    assert 'Table2' in sdf.tables

def test_add_table():
    sdf = PandaPack(tables=sample_table())
    new_table = SuperDataFrame(name='NewTable', table=sample_table())
    sdf.add_table(new_table)
    assert len(sdf.tables) == 2
    assert 'table_0' in sdf.tables
    assert 'NewTable' in sdf.tables
    
def test_add_dataframe():
    sdf = PandaPack(tables=sample_table())
    df=sample_table()
    sdf.add_table(df)
    assert len(sdf.tables) == 2
    assert 'table_0' in sdf.tables
    assert 'table_1' in sdf.tables

def test_get_table():
    table = SuperDataFrame(name='table_0', table=sample_table())
    sdf = PandaPack(tables=table)
    retrieved_table = sdf.get_table('table_0')
    assert retrieved_table.equals(table)

def test_get_table_names():
    table1 = SuperDataFrame(name='Table1', table=sample_table())
    table2 = SuperDataFrame(name='Table2', table=sample_table())
    sdf = PandaPack(tables=[table1, table2])
    names = sdf.get_table_names()
    assert 'Table1' in names
    assert 'Table2' in names

def test_add_foreign_key():
    table1 = SuperDataFrame(name='Table1', table=sample_table())
    table2 = SuperDataFrame(name='Table2', table=sample_table())
    pdp = PandaPack(tables=[table1, table2])
    pdp.add_foreign_key('Table1', 'A', 'Table2', 'B')
    assert len(table1.foreign_keys) == 1
    assert table2.foreign_keys == None  # No back-references are added

def test_to_and_from_disk(tmp_path):
    table = SuperDataFrame(name='table_0', table=sample_table())
    pdp = PandaPack(tables=table)
    file_path = tmp_path / "test_sdf.pkl"
    pdp.to_disk(file_path)
    
    loaded_pdp = PandaPack.from_disk(file_path)
    
    assert len(loaded_pdp.tables) == 1
    assert 'table_0' in loaded_pdp.tables
    assert loaded_pdp.get_table('table_0').equals(table)
    
def test_openai_client():
    config = SuperPandasConfig(llm_type='openai')
    table1 = SuperDataFrame(name='Table1', table=sample_table())
    table2 = SuperDataFrame(name='Table2', table=sample_table())
    pdp = PandaPack(tables=[table1, table2])
    pdp.add_foreign_key('Table1', 'A', 'Table2', 'B')
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
    table1 = SuperDataFrame(name='Table1', table=sample_table())
    table2 = SuperDataFrame(name='Table2', table=sample_table())
    pdp = PandaPack(tables=[table1, table2])
    pdp.add_foreign_key('Table1', 'A', 'Table2', 'B')
    spd = SuperPandas(pdp=pdp, config=config)
    response = spd.get_pdp_summary_from_llm()
    
    assert len(response) > 20 # The response should be a non-empty string
    assert 'Table1' in response  # The response should contain the name of the table
    assert 'Table2' in response  # The response should contain the name of the table
    assert 'A' in response  # The response should contain the name of the column
    assert 'B' in response  # The response should contain the name of the column
    assert 'int64' in response  # The response should contain the dtype of the column
    
def test_tgi_client_table_description():
    config = SuperPandasConfig(llm_type='tgi')
    table = SuperDataFrame(name='Table1', table=sample_table())
    pdp = PandaPack(tables=table)
    spd = SuperPandas(pdp=pdp, config=config)
    response = spd.get_table_description_from_llm('Table1')
    
    assert len(response) > 20 # The response should be a non-empty string
    assert 'Table1' in response  # The response should contain the name of the table
    assert 'A' in response  # The response should contain the name of the column
    assert 'B' in response  # The response should contain the name of the column
    assert '64' in response  # The response should contain the dtype of the column