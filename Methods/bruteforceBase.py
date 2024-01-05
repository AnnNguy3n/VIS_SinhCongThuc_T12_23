import numpy as np
import numba as nb
from Methods.base import Base, convert_arrF_to_strF
import queryFuncs as qf
import multiprocessing as mp
import time
import pandas as pd
import os
import sqlite3
import getValueFuncs as gvf
import json


@nb.njit
def encode_formula(f, len_):
    return f[0::2]*len_ + f[1::2]


def check_data_operands(op_name_1: dict, op_name_2: dict):
    if len(op_name_1) != len(op_name_2): return False

    op_1_keys = list(op_name_1.keys())
    op_2_keys = list(op_name_2.keys())
    for i in range(len(op_name_1)):
        if op_name_1[op_1_keys[i]] != op_name_2[op_2_keys[i]]:
            return False

    return True


def set_up(DATABASE_PATH, SAVE_TYPE, DATA_OR_PATH, LABEL, MAX_CYCLE, MIN_CYCLE, METHOD, FIELDS, MODE, DIV_WGT_BY_MC):
    folder_data = f"{DATABASE_PATH}/{LABEL}"
    os.makedirs(folder_data, exist_ok=True)

    if type(DATA_OR_PATH) == str:
        data = pd.read_excel(DATA_OR_PATH)
    else:
        data = DATA_OR_PATH.copy()

    data = data[data["TIME"] <= MAX_CYCLE]
    data = data[data["TIME"] >= MIN_CYCLE]
    if DIV_WGT_BY_MC:
        MARKET_CAP = np.array(data.pop("MARKET_CAP"))
    else:
        MARKET_CAP = 1.0

    base = Base(data)
    if not os.path.exists(folder_data + "/operand_names.json"):
        with open(folder_data + "/operand_names.json", "w") as fp:
            json.dump(base.operand_name, fp, indent=4)
        operand_name = base.operand_name
    else:
        with open(folder_data + "/operand_names.json", "r") as fp:
            operand_name = json.load(fp)

    if not check_data_operands(base.operand_name, operand_name):
        raise Exception("Sai data operands, kiem tra lai ten truong, thu tu cac truong trong data")

    folder_method = folder_data + f"/METHOD_{METHOD}"
    os.makedirs(folder_method, exist_ok=True)
    if SAVE_TYPE == 0:
        connection = sqlite3.connect(f"{folder_method}/f.db")
    else:
        connection = None

    if MODE == 0:
        list_gvf = [getattr(gvf, key) for key in FIELDS.keys()]
        list_field = []
        for key in FIELDS.keys():
            list_field += FIELDS[key]
    else:
        raise

    return data, connection, list_gvf, list_field, folder_method, MARKET_CAP


class BruteforceBase(Base):
    def __init__(self,
                 DATABASE_PATH,
                 SAVE_TYPE,
                 DATA_OR_PATH,
                 LABEL,
                 INTEREST,
                 NUM_CYCLE,
                 MAX_CYCLE,
                 MIN_CYCLE,
                 METHOD,
                 FIELDS,
                 MODE,
                 NUM_CHILD_PROCESS,
                 FILTERS,
                 DIV_WGT_BY_MC,
                 TARGET,
                 TMP_STRG_SIZE,
                 PERIODIC_SAVE_TIME):
        data, connection, list_gvf, list_field, main_folder, MARKET_CAP = set_up(
            DATABASE_PATH, SAVE_TYPE, DATA_OR_PATH, LABEL, MAX_CYCLE, MIN_CYCLE, METHOD, FIELDS, MODE, DIV_WGT_BY_MC
        )
        super().__init__(data)
        self.connection = connection
        self.list_gvf = list_gvf
        self.list_field = list_field
        self.main_folder = main_folder
        self.MARKET_CAP = MARKET_CAP
        self.storage_size = TMP_STRG_SIZE

        self.num_child_process = NUM_CHILD_PROCESS
        self.mode = MODE
        self.num_cycle = NUM_CYCLE
        self.filters = FILTERS
        self.save_type = SAVE_TYPE
        self.target = TARGET
        self.periodic_save_time = PERIODIC_SAVE_TIME
        self.interest = INTEREST

        self.num_data_operand = len(self.operand_name.keys())
        self.max_cycle = MAX_CYCLE
        self.time = time.time()
        self.start_time = self.time
        if self.save_type == 0:
            self.cursor = connection.cursor()

        self.temp_list_weight = []
        self.temp_list_formula = []
        self.list_data = []

        self.temp_df_save = []
        self.target_count = 0
        self.save_count = 0

    def add_to_temp_storage(self, weights, formulas):
        len_ = len(weights)
        self.temp_list_weight.extend(list(weights))
        self.temp_list_formula.extend(list(formulas))
        return len_

    def parallel_process(self):
        n = self.num_child_process + 1
        temp_index = np.linspace(0, len(self.temp_list_weight), n+1)
        temp_index[0] = 0
        temp_index[-1] = len(self.temp_list_weight)
        temp_index = temp_index.astype(int)
        list_args = []
        for i in range(n):
            args = (
                self.temp_list_weight[temp_index[i]:temp_index[i+1]],
                self.temp_list_formula[temp_index[i]:temp_index[i+1]],
                self.mode, self.num_data_operand, self.list_gvf, self.num_cycle,
                self.MARKET_CAP, self.save_type,
                self.INDEX, self.PROFIT, self.PROFIT_RANK, self.SYMBOL, self.interest
            )
            list_args.append(args)

        with mp.Pool(self.num_child_process) as pool:
            result_map = pool.map_async(handler_process, list_args[1:])
            result_0 = handler_process(list_args[0])
            self.list_data = result_0
            result = result_map.get()
            for list_ in result:
                for k in range(self.num_cycle):
                    self.list_data[k].extend(list_[k])

    def check_and_create_table(self, num_operand):
        if self.save_type == 0:
            self.cursor.execute(qf.get_list_table())
            list_table = [t_[0] for t_ in self.cursor.fetchall()]
            for cycle in range(self.max_cycle-self.num_cycle+1, self.max_cycle+1):
                if f"{cycle}_{num_operand}" not in list_table:
                    self.cursor.execute(qf.create_table(num_operand, self.list_field, cycle))
                    print("Create", f"{cycle}_{num_operand}")
            self.connection.commit()

    def save_history(self, flag=1):
        if self.save_type == 0:
            self.cursor.execute("SAVEPOINT my_savepoint;")
        try:
            if self.count[0] == 0:
                if self.save_type == 0:
                    self.cursor.execute("RELEASE my_savepoint;")
                return

            if self.mode == 0:
                self.parallel_process()

                for i in range(self.num_cycle):
                    list_data_cols = []
                    if self.save_type == 0:
                        for k in range(self.current_formula_length):
                            list_data_cols.append(f"E{k}")
                    else:
                        list_data_cols.append("formula")

                    list_data_cols += [f_[0] for f_ in self.list_field]
                    data = pd.DataFrame(self.list_data[i])
                    data.columns = list_data_cols
                    data.insert(loc=0, column="id", value=np.arange(self.start_id, self.start_id+self.count[0]))
                    for key, val in self.filters.items():
                        data = operator_mapping[val[0]](data, key, val[1])

                    self.target_count += len(data)
                    if self.save_type == 0:
                        self.cursor.execute(qf.insert_rows(
                            f"{self.max_cycle-self.num_cycle+1+i}_{self.current_formula_length}",
                            data.values.tolist()
                        ))
                    else:
                        data.insert(loc=0, column="cycle", value=self.max_cycle-self.num_cycle+1+i)
                        if len(data) > 0:
                            self.temp_df_save.append(data)

                self.list_data.clear()
                self.start_id += self.count[0]
                self.change_checkpoint()
                np.save(self.main_folder+"/checkpoint.npy", np.asanyarray(self.current, dtype=object), allow_pickle=True)
                if self.save_type == 0:
                    self.connection.commit()

                self.count[0] = 0
                self.temp_list_weight.clear()
                self.temp_list_formula.clear()
                time_ = time.time()
                print("Saved", time_ - self.time, self.current)
                self.time = time_
                if self.target_count >= self.target or self.time - self.start_time >= self.periodic_save_time or flag == 1:
                    if self.save_type == 0:
                        pass
                    else:
                        if len(self.temp_df_save) > 0:
                            pd.concat(self.temp_df_save).to_csv(
                                self.main_folder + f"/result_{self.save_count}.csv", index=False
                            )
                            self.save_count += 1
                            self.temp_df_save.clear()

                    if self.target_count >= self.target:
                        raise Exception("Đã sinh đủ số lượng")

                    if self.time - self.start_time >= self.periodic_save_time:
                        self.start_time = self.time
            else:
                raise

        except Exception as ex:
            if self.save_type == 0:
                self.cursor.execute("ROLLBACK TO my_savepoint;")
                self.cursor.execute("RELEASE my_savepoint;")

            print("Rolled Back")
            raise Exception("Rolled Back", ex)


def less_than(data, col, val):
    return data[data[col] < val]

def greater_than(data, col, val):
    return data[data[col] > val]

def equal_to(data, col, val):
    return data[data[col] == val]

def less_than_or_equal_to(data, col, val):
    return data[data[col] <= val]

def greater_than_or_equal_to(data, col, val):
    return data[data[col] >= val]

operator_mapping = {
    ">": greater_than,
    "<": less_than,
    "==": equal_to,
    ">=": greater_than_or_equal_to,
    "<=": less_than_or_equal_to,
}


def handler_process(args):
    list_weight = args[0]
    list_formula = args[1]
    mode = args[2]
    num_data_operand = args[3]
    list_gvf = args[4]
    num_cycle = args[5]
    MARKET_CAP = args[6]
    save_type = args[7]
    INDEX = args[8]
    PROFIT = args[9]
    PROFIT_RANK = args[10]
    SYMBOL = args[11]
    INTEREST = args[12]

    list_data = [[] for _ in range(num_cycle)]
    for k in range(len(list_weight)):
        weight = list_weight[k] / MARKET_CAP
        formula = list_formula[k]
        if mode == 0:
            if save_type == 0:
                list_value = list(encode_formula(formula, num_data_operand))
            else:
                list_value = [convert_arrF_to_strF(formula)]
        else:
            raise

        temp_list = [list_value.copy() for _ in range(num_cycle)]
        for func in list_gvf:
            result = func(weight, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, num_cycle)
            for i in range(num_cycle):
                rs = result[i]
                if type(rs) == tuple or type(rs) == list:
                    temp_list[i].extend(list(rs))
                else:
                    temp_list[i].extend([rs])

        for i in range(num_cycle):
            list_data[i].append(temp_list[i])

    return list_data
