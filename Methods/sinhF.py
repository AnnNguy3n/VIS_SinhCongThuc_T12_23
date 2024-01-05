import numpy as np
import numba as nb
from Methods.base import Base, convert_arrF_to_strF
import queryFuncs as qf
import time
import pandas as pd
import os
from Methods.bruteforceBase import set_up


@nb.njit
def decode_formula(f, len_):
    rs = np.full(len(f)*2, 0, dtype=int)
    rs[0::2] = f // len_
    rs[1::2] = f % len_
    return rs


class SinhF(Base):
    def __init__(self,
                 DATABASE_PATH,
                 SAVE_TYPE,
                 DATA_OR_PATH,
                 LABEL,
                 INTEREST,
                 MAX_CYCLE,
                 MIN_CYCLE,
                 FIELDS,
                 DIV_WGT_BY_MC,
                 PERIODIC_SAVE_TIME=1800):
        METHOD = 2
        MODE = 0
        data, connection, list_gvf, list_field, main_folder, MARKET_CAP = set_up(
            DATABASE_PATH, SAVE_TYPE, DATA_OR_PATH, LABEL, MAX_CYCLE, MIN_CYCLE, METHOD, FIELDS, MODE, DIV_WGT_BY_MC
        )
        super().__init__(data)
        self.connection = connection
        self.scoring_func = list_gvf[0]
        self.list_field = list_field
        self.main_folder = main_folder
        self.MARKET_CAP = MARKET_CAP

        self.save_type = SAVE_TYPE
        self.periodic_save_time = PERIODIC_SAVE_TIME
        self.interest = INTEREST
        self.num_data_operand = len(self.operand_name.keys())
        self.max_cycle = MAX_CYCLE
        self.start_time = time.time()
        if self.save_type == 0:
            self.cursor = connection.cursor()

        self.list_data = [[] for _ in range(100)]

    def generate(self):
        # Checkpoint
        try:
            self.checkpoint = np.load(
                self.main_folder + "/checkpoint.npy", allow_pickle=True
            )
        except:
            self.checkpoint = np.array([], dtype=int)

        weight = np.zeros(len(self.PROFIT))
        self.len_checkpoint = self.checkpoint.shape[0]
        self.history = self.checkpoint.copy()
        self.__fill_gm2__(np.array([], dtype=int), 0, weight, -1.7976931348623157e+308)
        self.save()

    def __fill_gm2__(self, f_:np.ndarray, gen:int, w_:np.ndarray, score:float):
        if gen == 100:
            return

        if time.time() - self.start_time >= self.periodic_save_time:
            self.history = f_
            self.save()
            self.start_time = time.time()
            print("Da luu")

        if self.len_checkpoint > gen and (self.checkpoint[:gen] == f_[:gen]).all():
            start = self.checkpoint[gen]
        else:
            start = 0

        formula = np.append(f_, start)

        if gen == 0:
            stop = 2*self.num_data_operand
        else:
            stop = 4*self.num_data_operand

        sub_list = []
        if gen > 0:
            pre = formula[gen-1]
            if pre < 2*self.num_data_operand:
                sub_list.append(
                    (pre+self.num_data_operand)%(2*self.num_data_operand)
                )
                for i in range(gen-2, -1, -1):
                    ele = formula[i]
                    if ele < 2*self.num_data_operand:
                        sub_list.append(
                            (ele+self.num_data_operand)%(2*self.num_data_operand)
                        )
                    else:
                        break
            else:
                sub_list.append(
                    (pre+self.num_data_operand)%(2*self.num_data_operand)
                    + 2*self.num_data_operand
                )
                for i in range(gen-2, -1, -1):
                    ele = formula[i]
                    if ele >= 2*self.num_data_operand:
                        sub_list.append(
                            (ele+self.num_data_operand)%(2*self.num_data_operand)
                            + 2*self.num_data_operand
                        )
                    else:
                        break

        for k in range(start, stop):
            if k in sub_list:
                continue

            formula[gen] = k
            operator = k // self.num_data_operand
            operand = k % self.num_data_operand
            if operator == 0:
                weight = w_ + self.OPERAND[operand]
            elif operator == 1:
                weight = w_ - self.OPERAND[operand]
            elif operator == 2:
                weight = w_ * self.OPERAND[operand]
            else:
                weight = w_ / self.OPERAND[operand]

            weight_ = weight.copy()
            weight_[np.isnan(weight_)] = -1.7976931348623157e+308
            weight_[np.isinf(weight_)] = -1.7976931348623157e+308
            cur_scr = self.scoring_func(weight_, self.INDEX, self.PROFIT, self.PROFIT_RANK, self.SYMBOL, self.interest, 1)
            if cur_scr > score:
                self.list_data[gen].append(
                    list(formula[:gen+1].copy())+[cur_scr]
                )
                self.__fill_gm2__(formula, gen+1, weight, cur_scr)
        
        self.history = formula

    def save(self):
        if self.save_type == 0:
            self.cursor.execute(qf.get_list_table())
            list_table = [t_[0] for t_ in self.cursor.fetchall()]

        for k in range(100):
            if len(self.list_data[k]) > 0:
                if self.save_type == 0:
                    if f"{self.max_cycle}_{k+1}" not in list_table:
                        self.cursor.execute(qf.create_table_sinhF(k+1, self.list_field, self.max_cycle))
                        self.connection.commit()
                else:
                    list_col = ["formula"] + [self.list_field[0][0]]
                    os.makedirs(self.main_folder + f"/Gen_{k+1}", exist_ok=True)

                if self.save_type == 0:
                    self.cursor.execute(qf.insert_rows(
                        f"{self.max_cycle}_{k+1}",
                        self.list_data[k]
                    ))
                else:
                    temp_data = []
                    for lst_ in self.list_data[k]:
                        fml = convert_arrF_to_strF(decode_formula(
                            np.array(lst_[:-1], int), self.num_data_operand
                        ).astype(int))
                        temp_data.append((fml, lst_[-1]))

                    data = pd.DataFrame(temp_data)
                    data.columns = list_col
                    data.to_csv(self.main_folder + f"/Gen_{k+1}/result_{self.start_time}.csv")

                self.list_data[k].clear()

        np.save(self.main_folder + "/checkpoint.npy", self.history, allow_pickle=True)
        if self.save_type == 0:
            self.connection.commit()
        self.start_time = time.time()
        print("Da luu")