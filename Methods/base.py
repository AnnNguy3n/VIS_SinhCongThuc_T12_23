import numpy as np
from pandas import DataFrame


class Base:
    def __init__(self, data: DataFrame) -> None:
        data = data.reset_index(drop=True)
        data.fillna(0.0, inplace=True)

        # Check cac cot bat buoc
        drop_cols = ["TIME", "PROFIT", "SYMBOL", "VALUEARG"]
        for col in drop_cols:
            if col not in data.columns:
                raise Exception(f"Thieu cot {col}")

        # Check dtype cua TIME va PROFIT
        if data["TIME"].dtype != "int64":
            dtype_ = data["TIME"].dtype
            raise Exception(f"dtype cua TIME khong phai int64 ({dtype_})")

        if data["PROFIT"].dtype != "float64":
            dtype_ = data["PROFIT"].dtype
            raise Exception(f"dtype cua PROFIT khong phai float64 ({dtype_})")

        # Check thu tu cot TIME
        if data["TIME"].diff().max() > 0:
            raise Exception("Cot TIME phai giam dan")

        # INDEX
        time_uni = data["TIME"].unique()
        index = []
        for i in range(data["TIME"].max(), data["TIME"].min()-1, -1):
            if i not in time_uni:
                raise Exception(f"Thieu chu ky {i}")

            index.append(data[data["TIME"]==i].index[0])

        index.append(data.shape[0])
        self.INDEX = np.array(index)

        # Loai cac cot co kieu du lieu khong phai int64 va float64
        for col in data.columns:
            if col not in drop_cols and data[col].dtype not in ["int64", "float64"]:
                drop_cols.append(col)

        self.drop_cols = drop_cols

        # Cac thuoc tinh
        self.data = data
        self.PROFIT = np.array(data["PROFIT"], float)
        self.PROFIT[self.PROFIT < 5e-324] = 5e-324

        operand_data = data.drop(columns=drop_cols)
        operand_name = operand_data.columns
        self.operand_name = {i:operand_name[i] for i in range(len(operand_name))}
        self.OPERAND = np.transpose(np.array(operand_data, float))

        symbol_name = data["SYMBOL"].unique()
        self.symbol_name = {symbol_name[i]:i for i in range(len(symbol_name))}
        self.SYMBOL = np.array([self.symbol_name[s] for s in data["SYMBOL"]])
        self.symbol_name = {v:k for k,v in self.symbol_name.items()}

        self.PROFIT_RANK = np.zeros(data.shape[0])
        for i in range(data["TIME"].min(), data["TIME"].max()+1):
            mask = data["TIME"] == i
            self.PROFIT_RANK[mask] = data.loc[mask, "PROFIT"].rank(method="min") / len(self.PROFIT_RANK[mask])


__STRING_OPERATOR = "+-*/"

def convert_arrF_to_strF(arrF):
    strF = ""
    for i in range(len(arrF)):
        if i % 2 == 1:
            strF += str(arrF[i])
        else:
            strF += __STRING_OPERATOR[arrF[i]]

    return strF

def convert_strF_to_arrF(strF):
    f_len = sum(strF.count(c) for c in __STRING_OPERATOR) * 2
    str_len = len(strF)
    arrF = np.full(f_len, 0)

    idx = 0
    for i in range(f_len):
        if i % 2 == 1:
            t_ = 0
            while True:
                t_ = 10*t_ + int(strF[idx])
                idx += 1
                if idx == str_len or strF[idx] in __STRING_OPERATOR:
                    break

            arrF[i] = t_
        else:
            arrF[i] = __STRING_OPERATOR.index(strF[idx])
            idx += 1

    return arrF
