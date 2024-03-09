
get_architecture_prompt = "Give a brief description in less than 100 words, of the architecture given in the following schema: 'table_name': {{'column_name': 'dtype', 'column_name': 'dtype', ...}}. Mention foreign key relationships if present.\n\nArchitecture: {arch}"

get_table_description_prompt = "Give a brief description in less than 50 words, of the table with the following schema: 'table_name': {{'column_name': 'dtype', 'column_name': 'dtype', ...}}.\n\nTable: {arch}"