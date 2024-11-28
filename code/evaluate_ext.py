MODULE='cgan_kd_sminebugfixed_keyS_errDbackward_kdData0924_WT2_T10_K32_BSZ50_new_interval'
EPOCH='bestg'
ATTACK=1

bptts=[80,70,60,50,40,30,20]
bpt=[4/bptt for bptt in bptts]
msgacc=[]
bitacc=[]
meteor=[]
spacymeteor=[]
sbert=[]
lmloss=[]
entail=[]
ss=[]

if ATTACK:
    cor_msgacc=[]
    cor_bitacc=[]
    cor_meteor=[]
    cor_spacymeteor=[]
    cor_sbert=[]
    cor_lmloss=[]
    cor_entail=[]
    cor_ss=[]
    diff_msgacc=[]
    diff_bitacc=[]
    diff_meteor=[]
    diff_spacymeteor=[]
    diff_sbert=[]
    diff_lmloss=[]
    diff_entail=[]
    diff_ss=[]


def findline(lines):
    for line in lines:
        if line.startswith('| test |'):
            return line

for bptt in bptts:
    f=open(f'evaluate_{MODULE}_e{EPOCH}_b{bptt}_jz.log')
    datas=findline(f.readlines()).split(' ')
    values=[float(data) for data in datas if data.split('.')[0].isdigit()]
    lmloss.append(values[0])
    msgacc.append(values[1])
    bitacc.append(values[2])
    meteor.append(values[3])
    spacymeteor.append(values[4])
    sbert.append(values[5])
    entail.append(values[6])
    ss.append(values[7])
    f.close()
    
    if ATTACK:
        # fatk=open(f'{MODULE}_eval_adapt_rob_wt2atk_seq{bptt}.log')
        # fatk=open(f'yprob_embdistort_distweight0.01_{MODULE}_eval_optimizingAtkDec_rob_seq{bptt}.log')
        fatk=open(f'1214initModified_yprob_Enc_CEdistort_distweight0_atkweight1_{MODULE}_eval_optimizingAtkEnc_rob_seq{bptt}.log')
        print(fatk)
        atk_datas=findline(fatk.readlines()).split(' ')
        atk_values=[float(data) for data in atk_datas if data.split('.')[0].isdigit()]
        cor_lmloss.append(atk_values[0])
        cor_msgacc.append(atk_values[1])
        cor_bitacc.append(atk_values[2])
        cor_meteor.append(atk_values[3])
        cor_spacymeteor.append(atk_values[4])
        cor_sbert.append(atk_values[5])
        cor_entail.append(atk_values[6]) #NOTICE!!!!
        cor_ss.append(atk_values[7])
        fatk.close()
        diff_lmloss.append(values[0]-atk_values[0])
        diff_msgacc.append(values[1]-atk_values[1])
        diff_bitacc.append(values[2]-atk_values[2])
        diff_meteor.append(values[3]-atk_values[3])
        diff_spacymeteor.append(values[4]-atk_values[4])
        diff_sbert.append(values[5]-atk_values[5])
        
print(MODULE,'= {')
# print('BPTT:',bptts)
print("    'bpw':",bpt,',')

# print("    'lmloss'",lmloss)
# print("'msgacc':",msgacc)
print("    'acc':",bitacc,',')
# print("'meteor:'",meteor)
print("    'meteor':",spacymeteor,',')
print("    'sbert':",sbert,',')
print("    'entail':",entail,',')
print("    'ss':",ss,',')

if ATTACK:
    # print('adapt attack lm loss',cor_lmloss)
    # print("'adapt attack msgacc:'",cor_msgacc)
    print("    'cor_acc':",cor_bitacc,',')
    # print('adapt attack meteor:',cor_meteor)
    print("    'cor_meteor':",cor_spacymeteor,',')
    print("    'cor_sbert':",cor_sbert,',')
    print("    'cor_entail':",cor_entail,',')
    print("    'cor_ss':",cor_ss,)
print('}')
        
        
        
