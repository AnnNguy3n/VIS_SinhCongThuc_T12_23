def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def create_table(len_formula, list_field, cycle):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(len_formula)]
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "{cycle}_{len_formula}" (
    "id" INTEGER NOT NULL,
    {temp.join(list_formula_col)}
    {temp.join(list_field_col)}
    PRIMARY KEY ("id")
)'''


def insert_rows(table_name, list_of_list_value):
    if len(list_of_list_value) == 0:
        return ";"
    
    temp_list = []
    for list_value in list_of_list_value:
        temp = ""
        for value in list_value:
            if type(value) == str:
                temp += f'"{value}",'
            else:
                temp += f"{value},"
        
        temp_list.append(f"({temp[:-1]}),")
    
    text = "".join(temp_list)

    return f'''INSERT INTO "{table_name}" VALUES {text[:-1]};'''
