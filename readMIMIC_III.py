import wfdb, os
import pprint
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from wfdb import processing


class readMIMIC_III():
    def __init__(self,file='RECORDS', save_dir = 'patient'):
        with open(file,'r') as f:
            self.dbs = [line.strip() for line in f.readlines()]
        self.Num_p , self.n= len(self.dbs) , 0
        self.Num_file = 0
        self.root_dbs = self.dbs[0][0:2]
        self.desired_signal = ['II','PLETH','ABP']
        fs , _duratian = 125 , 60
        self.sig_len = fs * _duratian
        if not os.path.exists(f'./{save_dir}'):
            os.mkdir(f'./{save_dir}')
        self.save_dir = save_dir

    def get_record(self,database, seg, patient_dir):
         if '_' in seg and len(seg.split('_')[1]) == 4:
            seg_header = wfdb.rdheader(seg, pn_dir = f'mimic3wdb/{self.root_dbs}/{database}')
            if seg_header.__dict__['sig_len'] > self.sig_len and all(elem in seg_header.__dict__['sig_name'] for elem in self.desired_signal):
            
            # print(f'There are the desired signals in the segment: {seg}')
            # print(f'fs = {seg_header.__dict__["fs"]}')
            # pprint.pprint(seg_header.__dict__['sig_name'], width= 1)
                record = wfdb.rdrecord(
                    seg,
                    pn_dir = f'mimic3wdb/{self.root_dbs}/{database}',
                    sampto = self.sig_len,
                    channel_names= self.desired_signal)

                df = record.to_dataframe()
                                                                    
                os.mkdir(f"./{self.save_dir}/{patient_dir}")
                df.to_csv(f'./{self.save_dir}/{patient_dir}/{seg}.csv')
                print(f'Number of Patient that was downloads : {len(os.listdir(f"./{self.save_dir}/"))}/{self.Num_desired_pat}')
                self.Num_file += 1
                print(f'File {seg} was downloaded')

    def run(self, Num_desired_pat = 40):
        self.Num_desired_pat = Num_desired_pat
        print(f'Number of patient: {self.Num_p}')
        for db in self.dbs:
            try:
                if len(os.listdir(f'./{self.save_dir}/')) < self.Num_desired_pat:
                    self.n+= 1
                    layout_hea = wfdb.rdheader(f'{db[:-1]}_layout', pn_dir=f'mimic3wdb/{self.root_dbs}/{db}')
                    # pprint.pprint(record1.__dict__['seg_name'], width= 1)  

                    if all(elem in layout_hea.__dict__['sig_name'] for elem in self.desired_signal):
                        print(f'{self.n}/{self.Num_p} \nThere are the desired signals in layout {db[:-1]}')

                        record_hea = wfdb.rdheader(f'{db[:-1]}', pn_dir=f'mimic3wdb/{self.root_dbs}/{db}')
                        
                        for seg in record_hea.__dict__['seg_name']:
                            pat_dir = seg.split('_')[0]
                            if not os.path.exists(f"./{self.save_dir}/{pat_dir}"):
                                self.get_record(database = db, seg = seg, patient_dir=pat_dir)
                               
            except Exception as e:
                print(f'Error in {db[:-1]} the error is:-->>\n{e}')

        print(f'Number of file was downloaded  {self.Num_file}')

    def plot_record(self, record_name, sig_start = 0, sig_end = None):
        record = wfdb.rdrecord(
                    record_name,
                    pn_dir = f'mimic3wdb/{record_name[0:2]}/{record_name.split("_")[0]}',
                    sampfrom = sig_start,
                    sampto = sig_end,
                    channel_names= self.desired_signal)

        record_hea = wfdb.rdheader(record_name, pn_dir=f'mimic3wdb/{record_name[0:2]}/{record_name.split("_")[0]}')

        pprint.pprint(f'fs = {record_hea.__dict__}',width=1)

        # wfdb.plot_wfdb(record)

        df = record.to_dataframe()
        
        # df.to_csv('./3199828_0003')
        
class read_csv():
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
    
    def run(self):
        columns = self.df.columns
        self.df = self.df.rename(columns={columns[0]:'Time'})
        # self.df['Time'] = self.df['Time'].str.split(':').str[-1]
        self.df['Time'] = self.df['Time'].apply(lambda x: x.split(':')[-1][0:-3])

        df_new = self.df[(self.df['ABP'] > 50) & (self.df['ABP'] < 180)]
        
        if df_new['ABP'].isnull().all() or df_new['ABP'].nunique() == 1:
            raise ValueError('There is not Signal ABP')

        if df_new['II'].isnull().all()  or df_new['II'].nunique() == 1:
            raise ValueError('There is not Signal ECG')

        if df_new['PLETH'].isnull().all()  or df_new['II'].nunique() == 1:
            raise ValueError('There Is not Signal PPG or it is constant')

        fig, axs = plt.subplots(3,1)
        self.df.plot(ax=axs[0],y='II')
        self.df.plot(ax=axs[1],y='ABP')
        self.df.plot(ax=axs[2],y='PLETH')
        axs[0].set_title(f'File name : {self.file_name}')
        plt.legend()
        plt.show()
        # plt.savefig(f'./onworking/plots/{self.file_name.split("/")[-1].split(".")[0]}.png')
        plt.close()
        

if __name__=='__main__':
    file = './RECORDS-33'
    data1 = readMIMIC_III(file = file, save_dir= 'patient-33')
    data1.run(Num_desired_pat = np.inf)
    # data1.plot_record(record_name='3199828_0003',sig_start=0 ,sig_end = 7500)
    

    # group = 32
    # list_file = os.listdir(f'./patient-{group}')
    # for pat in list_file:
    #     seg_list = os.listdir(f'./patient-{group}/{pat}')
    #     for seg in seg_list:
    #         file =f'./patient-{group}/{pat}/{seg}'
    #         data1 = read_csv(file_name = file)
    #         data1.run()
    # n = 1
    # list_file = os.listdir(f'./onworking')
    # for pat in list_file:
    #     data1 = read_csv(file_name = f'./onworking/{pat}')
    #     try:
    #         data1.run()
    #         n+=1
    #         print(pat)
    #     except Exception as e:
    #         print(e)
    # print(n)