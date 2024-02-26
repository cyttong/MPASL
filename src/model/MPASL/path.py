import os

class Path:
    def __init__(self, args):
        self.data = f'/home/henu/PycharmProjects/newcyt/MPASL-main/data/{args.dataset}/'
        self.misc = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/{args.dataset}/'
        self.emb = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/{args.dataset}/emb/'
        self.case_st = f'/home/henu/PycharmProjects/newcyt/MPASL-main/case_st/{args.dataset}/'
        self.output = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/no_sw'

        if args.abla_exp == True:
            self.output = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/abla_exp/'
        
        if args.top_k == True:
            self.output = f"{self.output}/topk/"

        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/no_sw/')

        self.check_dir(self.data)
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/')
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/case_st/')
        self.check_dir(self.case_st)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)

class Path_SW:
    def __init__(self, args):
        self.data = f'/home/henu/PycharmProjects/newcyt/MPASL-main/data/{args.dataset}/'
        self.misc = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/{args.dataset}/'
        self.emb = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/{args.dataset}/emb/'
        self.case_st = f'/home/henu/PycharmProjects/newcyt/MPASL-main/case_st/{args.dataset}/'
        self.output = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/sw/'

        if args.abla_exp == True:
            self.output = f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/abla_exp/'
        
        if args.top_k == True:
            self.output = f"{self.output}/topk/"

        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/')
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/output/{args.dataset}/MPASL_p0{args.p_hop}_h_{args.h_hop}/sw/')

        self.check_dir(self.data)
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/src/model/MPASL/misc/')
        self.check_dir(self.misc)
        self.check_dir(self.emb)
        self.check_dir(f'/home/henu/PycharmProjects/newcyt/MPASL-main/case_st/')
        self.check_dir(self.case_st)
        self.check_dir(self.output)

    def check_dir(self, p):
        if not os.path.isdir(p):
            os.mkdir(p)
