import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# from Methods.M0.generator import Generator
# vis = Generator(
#     DATABASE_PATH="/media/nguyen/VIS_DB/DBs",
#     SAVE_TYPE=0,
#     DATA_OR_PATH="/home/nguyen/Desktop/VIS_SinhCongThuc_T12_23/File_7_truong_HOSE_File3_2023_Field.xlsx",
#     LABEL="VN_7Fs",
#     INTEREST=1.06,
#     NUM_CYCLE=10,
#     MAX_CYCLE=2023,
#     MIN_CYCLE=2007,
#     FIELDS = {
#         "single_investment": [
#             ("GeoPro", "REAL"),
#             ("HarPro", "REAL"),
#             ("Value", "REAL"),
#             ("Profit", "REAL"),
#             ("ValGLim", "REAL"),
#             ("GeoLim", "REAL"),
#             ("ValHLim", "REAL"),
#             ("HarLim", "REAL"),
#             ("GeoRank", "REAL"),
#             ("HarRank", "REAL")
#         ],
#         "multi_investment": [
#             ("Nguong", "REAL"),
#             ("GeoNgn", "REAL"),
#             ("HarNgn", "REAL"),
#             ("ProNgn", "REAL")
#         ],
#         "multi_investment_strictly": [
#             ("Nguong2", "REAL"),
#             ("GeoNgn2", "REAL"),
#             ("HarNgn2", "REAL"),
#             ("ProNgn2", "REAL")
#         ],
#         "multi_investment_skip_20p_lowprofit": [
#             ("Nguong_20", "REAL"),
#             ("GeoNgn_20", "REAL"),
#             ("HarNgn_20", "REAL"),
#             ("ProNgn_20", "REAL")
#         ],
#         "multi_investment_skip_negative_profit": [
#             ("Nguong_snp", "REAL"),
#             ("GeoNgn_snp", "REAL"),
#             ("HarNgn_snp", "REAL"),
#             ("ProNgn_snp", "REAL")
#         ],
#     },
#     NUM_CHILD_PROCESS=7,
#     FILTERS={},
#     DIV_WGT_BY_MC=False,
#     TARGET=1000000000
# )
# vis.generate()


from Methods.M0.generator import Generator
vis = Generator(
    DATABASE_PATH="/media/nguyen/VIS_DB/DBs",
    SAVE_TYPE=0,
    DATA_OR_PATH="/home/nguyen/Desktop/VIS_SinhCongThuc_T12_23/HOSE_File3_2023_Field.xlsx",
    LABEL="VN_single_invest",
    INTEREST=1.06,
    NUM_CYCLE=10,
    MAX_CYCLE=2023,
    MIN_CYCLE=2007,
    FIELDS = {
        "single_investment": [
            ("GeoPro", "REAL"),
            ("HarPro", "REAL"),
            ("Value", "REAL"),
            ("Profit", "REAL"),
            ("ValGLim", "REAL"),
            ("GeoLim", "REAL"),
            ("ValHLim", "REAL"),
            ("HarLim", "REAL"),
            ("GeoRank", "REAL"),
            ("HarRank", "REAL")
        ],
        # "multi_investment": [
        #     ("Nguong", "REAL"),
        #     ("GeoNgn", "REAL"),
        #     ("HarNgn", "REAL"),
        #     ("ProNgn", "REAL")
        # ],
        # "multi_investment_strictly": [
        #     ("Nguong2", "REAL"),
        #     ("GeoNgn2", "REAL"),
        #     ("HarNgn2", "REAL"),
        #     ("ProNgn2", "REAL")
        # ],
        # "multi_investment_skip_20p_lowprofit": [
        #     ("Nguong_20", "REAL"),
        #     ("GeoNgn_20", "REAL"),
        #     ("HarNgn_20", "REAL"),
        #     ("ProNgn_20", "REAL")
        # ],
        # "multi_investment_skip_negative_profit": [
        #     ("Nguong_snp", "REAL"),
        #     ("GeoNgn_snp", "REAL"),
        #     ("HarNgn_snp", "REAL"),
        #     ("ProNgn_snp", "REAL")
        # ],
    },
    NUM_CHILD_PROCESS=7,
    FILTERS={
        "GeoLim": [">=", 1.35],
        "HarLim": [">=", 1.25]
    },
    DIV_WGT_BY_MC=False,
    TARGET=1000000000,
    TMP_STRG_SIZE=200000
)
vis.generate()