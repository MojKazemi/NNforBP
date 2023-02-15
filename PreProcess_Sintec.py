import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from SintecProj import SintecProj

class PreProcess_Sintec():

    def __init__(self,patient = '3400715'):
        self.fs = 125
        self.patient = patient
        self.regr_path = './Dataset/'
        self.df = pd.read_csv(self.regr_path +self.patient+'.csv').dropna()
        self.df.index = range(0, len(self.df))
        # Filtering the signal
        b, a = scipy.signal.butter(N=5, 
                                Wn=[1, 10], 
                                btype='band', 
                                analog=False,
                                output='ba',
                                fs=self.fs
                                )
        self.ecg_filt = scipy.signal.filtfilt(b, a, self.df['II'])
        self.ecg_diff = np.gradient(np.gradient(self.ecg_filt))
        self.ppg_filt = scipy.signal.filtfilt(b, a, self.df['PLETH'])
        self.Med_df_Out =pd.DataFrame()        

    def find_dbp_sbp(self):      
        # find DBP/SBP points
        DBPs, _ = scipy.signal.find_peaks(-self.df['ABP'], prominence=.5, distance=60, width=10)
        SBPs, _ = scipy.signal.find_peaks(self.df['ABP'], prominence=.5, distance=60, width=10)
        return DBPs,SBPs

    def ppg_feature(self):
        """ 
            Find featurs of PPG Signal: Peak(SP) - Trough(Tr) - Up Time - BTB Interval - PPG Height
        """

        # find SP peaks/time of Trough (min of PPG) and UpTime
        SPs, _ = scipy.signal.find_peaks(self.ppg_filt, prominence=.05, width=10)
        SP = SintecProj()
        SPs_new, [kde_ppg, kde_sp, x_ppg, min_] = SP.PPG_peaks_cleaner(self.ppg_filt, SPs)
        Trs, _ = scipy.signal.find_peaks(-self.ppg_filt, prominence=.05, width=30)
        # print(f'len Trs:{len(Trs)}')
        print(f'len SPs new: {len(SPs_new)}--- len Trs: {len(Trs)}')
        # Trs cleaning
        for index in range(min(len(Trs),len(SPs_new))):
            if Trs[index] > SPs_new[index]:
                if index ==0:
                    Trs = np.insert(Trs,index, 0)
                else:
                    tr_l,_ = scipy.signal.find_peaks(-self.ppg_filt[SPs_new[index-1]: SPs_new[index]], prominence=.05, width=10)
                    Trs = np.insert(Trs,index, SPs_new[index-1]+tr_l)

        # calculate UpTime
        UpTime , UpTime_f=[] , []
        dist={}

        for i in SPs_new:
            dist[i]=[]
            for j in Trs:
                dist[i].append(i-j)

        for sp,d in dist.items():
            try:
                uptime = min([i for i in d if 0<i<40])
            except:
                uptime =np.nan
            UpTime_f.append(uptime)

        # when there isn't uptime because couldn't find SP or Tr, mean of them will be replace by NaN
        mean_UpTime = int(np.round(np.mean([i for i in UpTime_f if not np.isnan(i)])))

        for i in UpTime_f:
            if np.isnan(i):
                i = mean_UpTime
            UpTime.append(i)

        # print(f'len UpTime : {len(UpTime)}')

        # calculate BTB for PPG
        BTB_ppg = []
        for index, i in enumerate(SPs_new):
            if index < (len(SPs_new)-1):
                BTB_ppg.append(SPs_new[index+1] - i)
        BTB_ppg.insert(0, round(np.mean(BTB_ppg[0:10])))

        # print(f'len BTB_ppg : {len(BTB_ppg)} - len SPs_new : {len(SPs_new)}')

        # calculate PPG height
        PPG_h = []
        loc_PPG_h = []
        for index,i in enumerate(Trs):
            try:
                if i<SPs_new[index]:
                    PPG_h.append(self.ppg_filt[SPs_new[index]]-self.ppg_filt[i])
                    loc_PPG_h.append(SPs_new[index])
            except:
                break

        # print(f'PPG_Height length: {len(PPG_h)}')
        return SPs_new, Trs, UpTime, PPG_h, BTB_ppg, loc_PPG_h

    def ecg_feature(self):
        """ 
            Find featurs of ECG Signal: R,T,P,Q,S series - BTB Interval
        """
        # find time of R,T,P,Q,S series
        Rs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=60)
        RTs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=20)
        RTPs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=10)
        T_LMin, _ = scipy.signal.find_peaks(-self.ecg_filt, prominence=.16, distance=10)  #.16

        BTB_R = []
        for index, i in enumerate(Rs):
            if index < (len(Rs)-1):
                BTB_R.append(Rs[index+1] - i)
        BTB_R.insert(0, round(np.mean(BTB_R[0:10])))

        Ts = [i for i in RTs if i not in Rs]
        Ps = [i for i in RTPs if i not in RTs]
        # print(f'len Ts {len(Ts)} - len RTs: {len(RTs)} - len Rs: {len(Rs)} - len RTPs:{len(RTPs)} - len Ps: {len(Ps)}')

        # define delta_T for each R_peak to find Q and S
        pre_i = 0
        Rs_dT = []
        for index, i in enumerate(Rs):
            if index == 0:
                delta_T = abs(i - Rs[index+1])/2
                Rs_dT.append((i, (0, i+delta_T)))
                pre_i = i

            elif index == len(Rs)-1:
                Rs_dT.append((i, (i-(i - pre_i)/2, len(self.ecg_filt))))

            else:
                Rs_dT.append((i, (i-(i - pre_i)/2, i+abs(i - Rs[index+1])/2)))
                pre_i = i
        # clean Ts, Ps      
        for rs,dt in Rs_dT:
            existP = self.existParam(Ts,lowLim=rs,upperLim=dt[1])
            if not existP:
                # print(type(self.ecg_filt))
                # print(rs,int(dt[1]))
                if int(dt[1])>len(self.ecg_filt):
                    _series = self.ecg_filt[rs:-1]
                else:
                    _series=self.ecg_filt[rs:int(dt[1])]
                Ts_new, _ = scipy.signal.find_peaks(_series, prominence=.01, distance=5)
                try:
                    Ts.append(rs+min(Ts_new))
                except:
                    pass
                # print(rs+Ts_new)

        # Find Qs, Ss
        Qs, Ss = [], []
        for i in T_LMin:
            for rs, dt in Rs_dT:
                if dt[0] <= i <= dt[1]:
                    if i < rs:
                        Qs.append(i)
                    else:
                        Ss.append(i)

        # print(f'Num of R_peaks: {len(Rs)}')
        # print(f'Num of P points: {len(Ps)}')
        # print(f'Num of Q points: {len(Qs)}')
        # print(f'Num of S points: {len(Ss)}')
        return Rs, Ts, Ps, Qs, Ss, BTB_R

    def existParam(self,_list,lowLim,upperLim):
        existP = False
        for param in _list:
            if lowLim< param<= upperLim:
                existP = True
                break
        return existP
        
    def Build_DataFrame(self):

        SPs_new, Trs, UpTime, PPG_h, BTB_ppg, loc_PPG_h = self.ppg_feature()
        Rs, Ts, Ps, Qs, Ss, BTB_R = self.ecg_feature()
        DBPs,SBPs = self.find_dbp_sbp()

        time = np.arange(0,(max(max(Rs), max(SPs_new), max(Ts), 
                                max(Qs), max(Ss), max(Ps),max(Trs))+1),1)
        real_time = time/self.fs
        # print(f'len real_time: {len(real_time)}')
        # print(f'max(Rs): {max(Rs)}\nmax(SPs_new): {max(SPs_new)}\nmax(T): {max(Ts)}')
        # Find Peak, Trough, UpTime, BTB of Peak and PPG height on PPG Signal
        df_Output = pd.DataFrame({'Time':real_time})
        df_Output.index = np.arange(len(real_time))
        df_Output['ppg_filt'] = self.ppg_filt[0:len(real_time)]
        df_Output.loc[Trs,'Tr'] = df_Output['ppg_filt'].iloc[Trs]
        df_Output['SPs_new'] = df_Output['ppg_filt'].iloc[SPs_new]
        df_Output.loc[SPs_new,'UpTime'] = [i/self.fs for i in UpTime]
        df_Output.loc[SPs_new,'BTB_PPG'] = [i/self.fs for i in BTB_ppg]

        df_Output.loc[loc_PPG_h,'PPG_h'] = PPG_h

        df_Output['DBP'] = self.df['ABP'].iloc[DBPs]
        df_Output['SBP']= self.df['ABP'].iloc[SBPs]

        # Find R,P,T,Q,S on ECG Signal
        df_Output['ecg_filt'] = self.ecg_filt[0:len(real_time)]
        df_Output['R'] = df_Output['ecg_filt'].iloc[Rs]
        df_Output.loc[Rs,'BTB_R']=[i/self.fs for i in BTB_R]
        df_Output['P'] = df_Output['ecg_filt'].iloc[Ps]
        df_Output['T'] = df_Output['ecg_filt'].iloc[Ts]
        df_Output['Q'] = df_Output['ecg_filt'].iloc[Qs]
        df_Output['S'] = df_Output['ecg_filt'].iloc[Ss]

        # Find PTT and HR
        SP = SintecProj()
        dataset = SP.find_PTT(self.ecg_filt,Rs,self.ppg_filt,SPs_new,self.patient)
        hr = dataset['HR'].values
        ptt = dataset['PTT'].values

        d =abs(len(dataset['HR'])-len(df_Output['ecg_filt']))
        if d != 0:
            # print(d)
            for i in range(d):
                hr = np.append(hr, np.nan)
                ptt= np.append(ptt, np.nan)

        df_Output['HR'] = hr
        df_Output['PTT'] = ptt
        return df_Output

    def df_median(self, df):
        '''
         Take Median from data feature
        '''
        df_output = pd.DataFrame()
        _list1 =[index for index,i in enumerate(df['HR'].notna().values) if i]
        _list1.insert(0,0)
        hr = df['HR']
        df = df.drop(columns=['HR'])
        for index,row in enumerate(_list1[1:]):
            df_output[row] = df.loc[_list1[index]:row].median()


        # for i in np.arange(0,len(df['Time']),step):
        #     if i +step > len(df['Time']):
        #         df_output[i+step] = df.loc[i:-1].median()
        #     else:
        #         df_output[i+step] = df.loc[i:i+step].median()

        df_output = df_output.T
        new_columns = []
        columns = df_output.columns
        for col in columns:
            if col == 'Unnamed: 0':
                new_columns.append('Median_row')
            else:
                new_columns.append('Med_'+col)

        df_output.columns = new_columns
        df_output['HR'] = hr
        df_output.to_csv('./asasa.csv')
        return df_output

    def plot_feature(self,df_Output):
        # plt.close('all')
        fig, axs = plt.subplots(2, 1,sharex=True)
        fig.set_size_inches(15,9)

        axs[0].plot(df_Output['ecg_filt'])
        axs[0].scatter(x=df_Output.index, y=df_Output['T'], c='y', s=40, label='T')
        # axs[0].scatter(x=Med_df_Out.index, y=Med_df_Out['Med_T'], marker='x', c='y', s=100, label='Med_T')
        axs[0].scatter(x=df_Output.index, y=df_Output['P'], c='c', s=40, label='P')
        # axs[0].scatter(x=Med_df_Out.index, y=Med_df_Out['Med_P'], marker='x', c='c', s=100, label='Med_P')
        axs[0].scatter(x=df_Output.index, y=df_Output['R'], c='r', s=40, label='R')
        # axs[0].scatter(x=Med_df_Out.index, y=Med_df_Out['Med_R'], marker='x', c='r', s=40, label='Med_R')
        axs[0].scatter(x=df_Output.index, y=df_Output['Q'], c='k', s=40, label='Q')
        # axs[0].scatter(x=Med_df_Out.index, y=Med_df_Out['Med_Q'], marker='x', c='k', s=40, label='Med_Q')
        axs[0].scatter(x=df_Output.index, y=df_Output['S'], c='m', s=40, label='S')
        # axs[0].scatter(x=Med_df_Out.index, y=Med_df_Out['Med_S'], marker='x', c='m', s=40, label='Med_S')
        axs[0].set_ylabel('ECG[(mV)]')

        df_Output['ppg_filt'].plot(ax=axs[1]) 
        axs[1].scatter(x=df_Output.index, y=df_Output['SPs_new'], c='y', s=40, label='SPs_new')
        axs[1].scatter(x=df_Output.index, y=df_Output['Tr'], c='r', s=40, label='Trough')
        axs[1].set_ylabel('PPG[]')

        x_ticks = np.arange(0, len(self.ppg_filt)+1, 500)
        for i in range(2):
            axs[i].set_xlabel('Time [s]')
            axs[i].set_xticks(x_ticks)
            axs[i].set_xticklabels((x_ticks/125).astype(int))

            axs[i].legend()
            axs[i].grid('both')
        plt.savefig(f'./Plots/LSTM/{self.patient}_features.png')
        # plt.show()

    def main(self):

        df_Output = self.Build_DataFrame()

        self.plot_feature(df_Output)

        # self.Med_df_Out = self.df_median(df_Output)

        self.Med_df_Out = df_Output.interpolate(method='polynomial',order=1)
        self.Med_df_Out = self.Med_df_Out

        self.Med_df_Out.to_csv('Dataset/NN_model/'+self.patient+'.csv')


        # print(self.Med_df_Out.columns)

if __name__=='__main__':
    # import os
    # file_list = os.listdir('./Dataset/')
    # for file in file_list[0:5]:
    #     if file.split('.')[1] == 'csv':
    #         print(patient:=file.split('.')[0])
    #         try:
    #             PrePS = PreProcess_Sintec(patient)
    #             PrePS.main()
    #         except:
    #             print(f"{patient} didn't complete")
    import os
    for n,file in enumerate(os.listdir('./Dataset/')):
        # pat_name = file.split('_')[0]
        if file.split('.')[1] == 'csv':
            print(patient:=file.split('.')[0])
            try:
                PrePS = PreProcess_Sintec(patient=patient)   #3600490   (3602521 shekam dard)
                PrePS.main()
            except:
                print(f"{patient} didn't complete")
    # print(len(PrePS.Med_df_Out))