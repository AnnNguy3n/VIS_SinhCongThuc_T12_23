import numpy as np
import numba as nb


@nb.njit
def geomean(arr):
    log_sum = sum(np.log(arr))
    return np.exp(log_sum/len(arr))

@nb.njit
def harmean(arr):
    dnmntor = sum(1.0/arr)
    return len(arr)/dnmntor


@nb.njit
def single_investment(WEIGHT, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, NUM_CYCLE):
    """
    Output: GeoPro, HarPro, Value, Profit, ValGLim, GeoLim, ValHLim, HarLim, GeoRank, HarRank
    """
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size)
    arr_ivalue = np.zeros(size)
    arr_invest_index = np.full(size, 0)
    for i in range(size-1, -1, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end]
        arr_max = np.where(wgt_==max(wgt_))[0]
        if arr_max.shape[0] == 1:
            arr_profit[idx] = PROFIT[start:end][arr_max[0]]
            arr_ivalue[idx] = wgt_[arr_max[0]]
            arr_invest_index[idx] = arr_max[0] + start
        else:
            arr_profit[idx] = INTEREST
            arr_ivalue[idx] = 1.7976931348623157e+308
            arr_invest_index[idx] = -1

    arr_rank_value = PROFIT_RANK[arr_invest_index]
    arr_rank_value[arr_invest_index==-1] = 0.5
    GeoRank = np.zeros(NUM_CYCLE)
    HarRank = np.zeros(NUM_CYCLE)
    GeoRank[0] = sum(np.log(arr_rank_value[:-NUM_CYCLE]))
    HarRank[0] = sum(1.0/arr_rank_value[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        r = arr_rank_value[-NUM_CYCLE+i]
        GeoRank[i+1] = GeoRank[i] + np.log(r)
        HarRank[i+1] = HarRank[i] + 1.0 / r

    Value = arr_ivalue[-NUM_CYCLE:]
    Profit = arr_profit[-NUM_CYCLE:]

    GeoPro = np.zeros(NUM_CYCLE)
    HarPro = np.zeros(NUM_CYCLE)
    GeoPro[0] = sum(np.log(arr_profit[:-NUM_CYCLE]))
    HarPro[0] = sum(1.0/arr_profit[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        p = arr_profit[-NUM_CYCLE+i]
        GeoPro[i+1] = GeoPro[i] + np.log(p)
        HarPro[i+1] = HarPro[i] + 1.0 / p

    GeoLim = GeoPro.copy()
    HarLim = HarPro.copy()
    ValGLim = np.zeros(NUM_CYCLE)
    ValGLim[0] = min(arr_ivalue[:-NUM_CYCLE])
    for i in range(NUM_CYCLE-1):
        ValGLim[i+1] = min(ValGLim[i], arr_ivalue[-NUM_CYCLE+i])

    ValGLim -= np.maximum(np.abs(ValGLim)*1e-9, 1e-9)
    ValHLim = ValGLim.copy()
    for v in arr_ivalue[:-NUM_CYCLE]:
        temp_profit = np.where(arr_ivalue > v, arr_profit, INTEREST)
        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE-1]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE-1])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE-1+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if temp_log_sum > GeoLim[i]:
                GeoLim[i] = temp_log_sum
                ValGLim[i] = v

            if temp_dnmntor < HarLim[i]:
                HarLim[i] = temp_dnmntor
                ValHLim[i] = v

    add_id = 0
    for k in range(-NUM_CYCLE, -1):
        add_id += 1
        v = arr_ivalue[k]
        temp_profit = np.where(arr_ivalue > v, arr_profit, INTEREST)
        temp_log_sum = sum(np.log(temp_profit[:k]))
        temp_dnmntor = sum(1.0/temp_profit[:k])
        for i in range(-1-k):
            p = temp_profit[k+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            idx_ = add_id + i
            if temp_log_sum > GeoLim[idx_]:
                GeoLim[idx_] = temp_log_sum
                ValGLim[idx_] = v

            if temp_dnmntor < HarLim[idx_]:
                HarLim[idx_] = temp_dnmntor
                ValHLim[idx_] = v

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i
        result = [
            np.exp(GeoPro[i]/n),
            n / HarPro[i],
            Value[i],
            Profit[i],
            ValGLim[i],
            np.exp(GeoLim[i]/n),
            ValHLim[i],
            n / HarLim[i],
            np.exp(GeoRank[i]/n),
            n / HarRank[i],
        ]
        results.append(result)

    return results


@nb.njit
def multi_investment(WEIGHT, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, NUM_CYCLE, n_val_per_cyc=5):
    """
    Output: Nguong, GeoNgn, HarNgn, ProNgn
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*n_val_per_cyc)
    for i in range(size-1, 0, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[n_val_per_cyc*idx:n_val_per_cyc*(idx+1)] = wgt_[:n_val_per_cyc]

    temp_arr_loop = np.unique(arr_loop[:n_val_per_cyc*(-NUM_CYCLE+1)])
    Nguong = np.zeros(NUM_CYCLE)
    GeoNgn = np.zeros(NUM_CYCLE)
    HarNgn = np.zeros(NUM_CYCLE)
    temp_profit = np.zeros(size-1)
    for ii in range(len(temp_arr_loop)):
        v = temp_arr_loop[ii]
        bool_wgt = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(size-1, 0, -1):
            idx = size - 1 - i
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(bool_wgt[start:end]) == 0:
                temp_profit[idx] = INTEREST
            else:
                temp_profit[idx] = PROFIT[start:end][bool_wgt[start:end]].mean()

        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if ii == 0:
                Nguong[i] = v
                GeoNgn[i] = temp_log_sum
                HarNgn[i] = temp_dnmntor
            else:
                if temp_log_sum > GeoNgn[i]:
                    Nguong[i] = v
                    GeoNgn[i] = temp_log_sum
                    HarNgn[i] = temp_dnmntor

    add_id = 0
    for k in range(-NUM_CYCLE+1, 0):
        add_id += 1
        if k == -1:
            add_val_loop = np.unique(arr_loop[-n_val_per_cyc:])
        else:
            add_val_loop = np.unique(arr_loop[k*n_val_per_cyc:(k+1)*n_val_per_cyc])

        # add_val_loop = np.setdiff1d(add_val_loop, temp_arr_loop) # Can't njit numba
        add_val_loop = np.array([x for x in add_val_loop if x not in temp_arr_loop])
        temp_arr_loop = np.append(temp_arr_loop, add_val_loop)
        for v in add_val_loop:
            bool_wgt = WEIGHT > v
            temp_profit[:] = 0.0
            for i in range(size-1, 0, -1):
                idx = size - 1 - i
                start, end = INDEX[i], INDEX[i+1]
                if np.count_nonzero(bool_wgt[start:end]) == 0:
                    temp_profit[idx] = INTEREST
                else:
                    temp_profit[idx] = PROFIT[start:end][bool_wgt[start:end]].mean()

            temp_log_sum = sum(np.log(temp_profit[:k]))
            temp_dnmntor = sum(1.0/temp_profit[:k])
            for i in range(-k):
                p = temp_profit[k+i]
                temp_log_sum += np.log(p)
                temp_dnmntor += 1.0 / p
                idx_ = add_id + i
                if temp_log_sum > GeoNgn[idx_]:
                    Nguong[idx_] = v
                    GeoNgn[idx_] = temp_log_sum
                    HarNgn[idx_] = temp_dnmntor

    ProNgn = np.zeros(NUM_CYCLE)
    for i in range(NUM_CYCLE-1, -1, -1):
        idx = NUM_CYCLE - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        v = Nguong[idx]
        mask_ = WEIGHT[start:end] > v
        if np.count_nonzero(mask_) == 0.0:
            ProNgn[idx] = INTEREST
        else:
            ProNgn[idx] = PROFIT[start:end][mask_].mean()

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i
        result = [
            Nguong[i],
            np.exp(GeoNgn[i]/n),
            n / HarNgn[i],
            ProNgn[i]
        ]
        results.append(result)

    return results


@nb.njit
def multi_investment_strictly(WEIGHT, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, NUM_CYCLE, n_val_per_cyc=5):
    """
    Output: Nguong2, GeoNgn2, HarNgn2, ProNgn2
    """
    size = INDEX.shape[0] - 1
    arr_loop = np.zeros((size-1)*n_val_per_cyc)
    for i in range(size-1, 0, -1):
        idx = size - 1 - i
        start, end = INDEX[i], INDEX[i+1]
        wgt_ = WEIGHT[start:end].copy()
        wgt_[::-1].sort()
        arr_loop[n_val_per_cyc*idx:n_val_per_cyc*(idx+1)] = wgt_[:n_val_per_cyc]

    temp_arr_loop = np.unique(arr_loop[:n_val_per_cyc*(-NUM_CYCLE+1)])
    Nguong2 = np.zeros(NUM_CYCLE)
    GeoNgn2 = np.zeros(NUM_CYCLE)
    HarNgn2 = np.zeros(NUM_CYCLE)
    ProNgn2 = np.zeros(NUM_CYCLE)
    temp_profit = np.zeros(size-1)
    for ii in range(len(temp_arr_loop)):
        v = temp_arr_loop[ii]
        temp_profit[:] = 0.0
        reason = 0
        bool_wgt = WEIGHT > v
        for i in range(size-2, -1, -1):
            start, end = INDEX[i], INDEX[i+1]
            inv_cyc_val = bool_wgt[start:end]
            if reason == 0:
                inv_cyc_sym = SYMBOL[start:end]
                end2 = INDEX[i+2]
                pre_cyc_val = bool_wgt[end:end2]
                pre_cyc_sym = SYMBOL[end:end2]
                coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                isin = np.full(end-start, False)
                for j in range(end-start):
                    if inv_cyc_sym[j] in coms:
                        isin[j] = True
                lst_pro = PROFIT[start:end][isin]
            else:
                lst_pro = PROFIT[start:end][inv_cyc_val]

            _idx_ = size - 2 - i
            if len(lst_pro) == 0:
                temp_profit[_idx_] = INTEREST
                if np.count_nonzero(inv_cyc_val) == 0:
                    reason = 1
            else:
                temp_profit[_idx_] = np.mean(lst_pro)
                reason = 0

        temp_log_sum = sum(np.log(temp_profit[:-NUM_CYCLE-1]))
        temp_dnmntor = sum(1.0/temp_profit[:-NUM_CYCLE-1])
        for i in range(NUM_CYCLE):
            p = temp_profit[-NUM_CYCLE-1+i]
            temp_log_sum += np.log(p)
            temp_dnmntor += 1.0 / p
            if ii == 0:
                Nguong2[i] = v
                GeoNgn2[i] = temp_log_sum
                HarNgn2[i] = temp_dnmntor
                ProNgn2[i] = temp_profit[-NUM_CYCLE+i]
            else:
                if temp_log_sum > GeoNgn2[i]:
                    Nguong2[i] = v
                    GeoNgn2[i] = temp_log_sum
                    HarNgn2[i] = temp_dnmntor
                    ProNgn2[i] = temp_profit[-NUM_CYCLE+i]

    add_id = 0
    for k in range(-NUM_CYCLE+1, 0):
        add_id += 1
        if k == -1:
            add_val_loop = np.unique(arr_loop[-n_val_per_cyc:])
        else:
            add_val_loop = np.unique(arr_loop[k*n_val_per_cyc:(k+1)*n_val_per_cyc])

        # add_val_loop = np.setdiff1d(add_val_loop, temp_arr_loop) # Can't njit numba
        add_val_loop = np.array([x for x in add_val_loop if x not in temp_arr_loop])
        temp_arr_loop = np.append(temp_arr_loop, add_val_loop)
        for v in add_val_loop:
            temp_profit[:] = 0.0
            reason = 0
            bool_wgt = WEIGHT > v
            for i in range(size-2, -1, -1):
                start, end = INDEX[i], INDEX[i+1]
                inv_cyc_val = bool_wgt[start:end]
                if reason == 0:
                    inv_cyc_sym = SYMBOL[start:end]
                    end2 = INDEX[i+2]
                    pre_cyc_val = bool_wgt[end:end2]
                    pre_cyc_sym = SYMBOL[end:end2]
                    coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                    isin = np.full(end-start, False)
                    for j in range(end-start):
                        if inv_cyc_sym[j] in coms:
                            isin[j] = True
                    lst_pro = PROFIT[start:end][isin]
                else:
                    lst_pro = PROFIT[start:end][inv_cyc_val]

                _idx_ = size - 2 - i
                if len(lst_pro) == 0:
                    temp_profit[_idx_] = INTEREST
                    if np.count_nonzero(inv_cyc_val) == 0:
                        reason = 1
                else:
                    temp_profit[_idx_] = np.mean(lst_pro)
                    reason = 0

            temp_log_sum = sum(np.log(temp_profit[:k-1]))
            temp_dnmntor = sum(1.0/temp_profit[:k-1])
            for i in range(-k):
                p = temp_profit[k-1+i]
                temp_log_sum += np.log(p)
                temp_dnmntor += 1.0 / p
                idx_ = add_id + i
                if temp_log_sum > GeoNgn2[idx_]:
                    Nguong2[idx_] = v
                    GeoNgn2[idx_] = temp_log_sum
                    HarNgn2[idx_] = temp_dnmntor
                    ProNgn2[idx_] = temp_profit[k+i]

    results = []
    for i in range(NUM_CYCLE):
        n = size - NUM_CYCLE + i - 1
        result = [
            Nguong2[i],
            np.exp(GeoNgn2[i]/n),
            n / HarNgn2[i],
            ProNgn2[i]
        ]
        results.append(result)

    return results


@nb.njit
def multi_investment_skip_20p_lowprofit(WEIGHT, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, NUM_CYCLE):
    """
    Output: Nguong_20, GeoNgn_20, HarNgn_20, ProNgn_20
    """
    size = INDEX.shape[0] - 1
    Nguong_20 = np.zeros(NUM_CYCLE)
    temp_nguong = -1.7976931348623157e+308
    for i in range(size-1, NUM_CYCLE-1, -1):
        start, end = INDEX[i], INDEX[i+1]
        values = WEIGHT[start:end]
        arrPro = PROFIT[start:end]
        mask = np.argsort(arrPro)
        n = int(np.ceil(float(len(mask)) / 5))
        ngn = np.max(values[mask[:n]])
        if ngn > temp_nguong:
            temp_nguong = ngn

    Nguong_20[0] = temp_nguong
    for i in range(NUM_CYCLE-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        values = WEIGHT[start:end]
        arrPro = PROFIT[start:end]
        mask = np.argsort(arrPro)
        n = int(np.ceil(float(len(mask)) / 5))
        ngn = np.max(values[mask[:n]])
        if ngn > temp_nguong:
            temp_nguong = ngn

        idx = NUM_CYCLE - i
        Nguong_20[idx] = temp_nguong

    GeoNgn_20 = np.zeros(NUM_CYCLE)
    HarNgn_20 = np.zeros(NUM_CYCLE)
    ProNgn_20 = np.zeros(NUM_CYCLE)
    for i in range(NUM_CYCLE):
        idx = NUM_CYCLE - 1 - i
        v = Nguong_20[idx]
        temp_profit = np.zeros(size-i)
        bool_wgt = WEIGHT > v
        for k in range(size-1, i-1, -1):
            _idx_ = size - 1 - k
            start, end = INDEX[k], INDEX[k+1]
            mask_ = bool_wgt[start:end]
            if np.count_nonzero(mask_) == 0:
                temp_profit[_idx_] = INTEREST
            else:
                temp_profit[_idx_] = PROFIT[start:end][mask_].mean()

        GeoNgn_20[idx] = geomean(temp_profit[:-1])
        HarNgn_20[idx] = harmean(temp_profit[:-1])
        ProNgn_20[idx] = temp_profit[-1]

    results = []
    for i in range(NUM_CYCLE):
        result = [
            Nguong_20[i],
            GeoNgn_20[i],
            HarNgn_20[i],
            ProNgn_20[i]
        ]
        results.append(result)

    return results


@nb.njit
def multi_investment_skip_negative_profit(WEIGHT, INDEX, PROFIT, PROFIT_RANK, SYMBOL, INTEREST, NUM_CYCLE):
    """
    Output: Nguong_snp, GeoNgn_snp, HarNgn_snp, ProNgn_snp
    """
    size = INDEX.shape[0] - 1
    Nguong_snp = np.zeros(NUM_CYCLE)
    start = INDEX[NUM_CYCLE]
    mask = PROFIT[start:] < 1.0
    Nguong_snp[0] = np.max(WEIGHT[start:][mask])

    for i in range(NUM_CYCLE-1, 0, -1):
        idx = NUM_CYCLE - i
        start, end = INDEX[i], INDEX[i+1]
        mask = PROFIT[start:end] < 1.0
        temp_nguong = max(WEIGHT[start:end][mask])
        temp_nguong = max(temp_nguong, Nguong_snp[idx-1])
        Nguong_snp[idx] = temp_nguong

    GeoNgn_snp = np.zeros(NUM_CYCLE)
    HarNgn_snp = np.zeros(NUM_CYCLE)
    ProNgn_snp = np.zeros(NUM_CYCLE)
    for i in range(NUM_CYCLE):
        idx = NUM_CYCLE - 1 - i
        v = Nguong_snp[idx]
        temp_profit = np.zeros(size-i)
        bool_wgt = WEIGHT > v
        for k in range(size-1, i-1, -1):
            _idx_ = size - 1 - k
            start, end = INDEX[k], INDEX[k+1]
            mask_ = bool_wgt[start:end]
            if np.count_nonzero(mask_) == 0:
                temp_profit[_idx_] = INTEREST
            else:
                temp_profit[_idx_] = PROFIT[start:end][mask_].mean()

        GeoNgn_snp[idx] = geomean(temp_profit[:-1])
        HarNgn_snp[idx] = harmean(temp_profit[:-1])
        ProNgn_snp[idx] = temp_profit[-1]

    results = []
    for i in range(NUM_CYCLE):
        result = [
            Nguong_snp[i],
            GeoNgn_snp[i],
            HarNgn_snp[i],
            ProNgn_snp[i]
        ]
        results.append(result)

    return results
