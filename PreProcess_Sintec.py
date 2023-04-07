import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os


from SintecProj import SintecProj

class PreProcess_Sintec():

    def __init__(self,file = '3400715_pat.csv'):
        self.showplot = True  
        self.fs = 125
        self.figsize = (15,9)
        # self.patient = patient
        self.patient_path = './Patients'
        self.preprocess_data(file)
        self.check_dir()
        self.plt_Feat_path = './Plots/Features'
        self.regr_path = './Dataset'

    def read_data(self,file):
        pat_name = file.split('_')[0]
        if file.split('_')[1] == 'pat.csv':
            df = pd.read_csv(f'{self.patient_path}/{file}',quotechar="'",sep=',',skiprows=[1])
            if df.iloc[0][0][0] == '"':
                df.columns = [x.replace('"',"") for x in df.columns]
                df.columns = [x.replace("'","") for x in df.columns]

                df['Time'] = df['Time'].apply(lambda x: x[3:-2])
                df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: x[:-1])
                df = df.replace('-', np.nan)	
                df.index = df['Time']

                def_columns = []
                for x in df.columns:
                    if 'ABP' in x or x=='II' or 'PLETH' in x:
                        def_columns.append(x)
                df = df[def_columns]
                df = df.astype(float)
            else:
                df['Time'] = df['Time'].apply(lambda x: x[1:-1])
                df.index = df['Time']
                df = df.replace('-', np.nan)
                df = df[['ABP','PLETH','II']]
                df = df.astype(float)
            if pat_name == '3601980': df = df[0:5021]
        else:
            df = pd.read_csv(f'{self.patient_path}/{file}')
            columns = df.columns
            df = df.rename(columns={columns[0]:'Time'})
            df['Time'] = df['Time'].apply(lambda x: x.split(':')[-1][0:-3])
            df = df.replace('', np.nan)
            df = df.astype(float)
            pat_name = pat_name + file.split('_')[1][0:4]
        
        df = self.cleaning_data(df)
        return df, pat_name
        
    def cleaning_data(self, df):

        df_new = df[(df['ABP'] > 50) & (df['ABP'] < 180)]
        
        if df_new['ABP'].isnull().all() or df_new['ABP'].nunique() == 1:
            raise ValueError('There is not Signal ABP or the signal is constant')

        if df_new['II'].isnull().all()  or df_new['II'].nunique() == 1:
            raise ValueError('There is not Signal ECG or the signal is constant')

        if df_new['PLETH'].isnull().all()  or df_new['PLETH'].nunique() == 1:
            raise ValueError('There Is not Signal PPG or or the signal is constant')

        # drop_rows = []
        # for i in range(8,len(df)):
        #     # Check if any column has too many consecutive equal values
        #     for col in df.columns:
        #         if df.loc[i-8:i,col].eq(df.loc[i,col]).all():
        #             drop_rows.append(i)

        #             break  
        # # print(f'drop row = {drop_row}')
        # df.drop(drop_rows, inplace=True)
        # df = df.reset_index(drop=True)

        fig, axs = plt.subplots(3, 1)
        df.plot(ax=axs[0],y='II')
        df.plot(ax=axs[1],y='ABP')
        df.plot(ax=axs[2],y='PLETH')

        if self.showplot: plt.show()
        
        return df_new
        
    def preprocess_data(self, file):
        self.df, self.patient = self.read_data(file)
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
           
        if np.isnan(self.ppg_filt).all():
            raise ValueError('After filtering the PPG signal got Nan') 
        
        if np.isnan(self.ecg_filt).all():
            raise ValueError('After filtering the ECG signal got Nan')

    def check_dir(self): 
        if not os.path.exists('./Dataset'):
            os.mkdir('./Dataset')
        if not os.path.exists('./Plots'):
            os.mkdir('./Plots')
        if not os.path.exists('./Plots/Features'):
            os.mkdir('./Plots/Features')   
    
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
        Trs, _ = scipy.signal.find_peaks(-self.ppg_filt, prominence=.05)
        # print(f'len Trs:{len(Trs)}')
        # print(f'len SPs new: {len(SPs_new)}--- len Trs: {len(Trs)}')
        # Trs cleaning
        for i in range(len(SPs_new)-1):
            elements = [x for x in Trs if SPs_new[i] < x < SPs_new[i+1]]
            if len(elements) > 1:
                elements.remove(max(elements))
                Trs = np.setdiff1d(Trs, elements)
            elif len(elements)==0:
                tr_l,_ = scipy.signal.find_peaks(-self.ppg_filt[SPs_new[i]: SPs_new[i+1]])
                Trs.append(SPs_new[i]+max(tr_l))


        for index in range(min(len(Trs),len(SPs_new))):
            if Trs[index] > SPs_new[index]:
                if index ==0:
                    Trs = np.insert(Trs,index, 0)
                else:
                    tr_l,_ = scipy.signal.find_peaks(-self.ppg_filt[SPs_new[index-1]: SPs_new[index]])
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

        # for i in Trs:
        #     min_greater,_index = self.find_smallest_greater(SPs_new,i)
        #     if _index == None:
        #         pass
        #     else:
        #         PPG_h.append(self.ppg_filt[min_greater]-self.ppg_filt[i])
        #         loc_PPG_h.append(_index)

        # print(f'PPG_Height length: {len(PPG_h)}')
        return SPs_new, Trs, UpTime, PPG_h, BTB_ppg, loc_PPG_h

    def find_smallest_greater(self, arr, i):
        min_val, min_index = None, None
        for index, val in enumerate(arr):
            if val > i and (min_val is None or val < min_val):
                min_val = val
                min_index = index
        return min_val, min_index

    def ecg_feature(self):
        """ 
            Find featurs of ECG Signal: R,T,P,Q,S series - BTB Interval
        """
        # find time of R,T,P,Q,S series
        Rs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=60)
        RTs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=20)
        RTPs, _ = scipy.signal.find_peaks(self.ecg_filt, prominence=.05, distance=10)
        T_LMin, _ = scipy.signal.find_peaks(-self.ecg_filt, prominence=.16, distance=10)  #.16
        
        plt.figure()
        plt.plot(self.ecg_filt)
        plt.plot(Rs,self.ecg_filt[Rs], 'o')
        if self.showplot: plt.show()
        BTB_R = []
        for index, i in enumerate(Rs):
            if index < (len(Rs)-1):
                BTB_R.append(Rs[index+1] - i)
        try:
            BTB_R.insert(0, round(np.mean(BTB_R[0:10])))
        except:
            print(f'dddd{RTPs}')

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
        # print(f'T_LMin{len(T_LMin)}')     
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
            existP = self.existParam(Ps,lowLim=dt[0],upperLim = rs)
            if not existP:
                _series=self.ecg_filt[int(dt[0]):rs]
                Ps_new, _ = scipy.signal.find_peaks(_series, prominence=.01, distance=5)
                try:
                    Ps.append(dt[0]+max(Ps_new))
                except:
                    pass


        # Find Qs, Ss
        Qs, Ss = [], []
        for i in T_LMin:
            for rs, dt in Rs_dT:
                if dt[0] <= i <= dt[1]:
                    if i < rs:
                        Qs.append(i)
                    else:
                        Ss.append(i)

        for rs,dt in Rs_dT:
            exist_Qs = self.existParam(Qs,lowLim=dt[0],upperLim=rs)
            if not exist_Qs:
                _series=self.ecg_filt[int(dt[0]):rs]
                q_min, _ = scipy.signal.find_peaks(-_series, prominence=.01, distance=1)
                try:
                    Qs.append(dt[0] + max(q_min))
                except Exception as e:
                    # print('fall', e)
                    pass
            exist_ss = self.existParam(Ss,lowLim=rs,upperLim=dt[1])
            if not exist_ss:
                _series = self.ecg_filt[rs:int(dt[1])]
                s_min,_ = scipy.signal.find_peaks(-_series, prominence=.01, distance=5)
                try:
                    Ss.append(rs + min(s_min))
                except Exception as e:
                    # print('fall', e)
                    pass
                
        # print(f'len ecg {len(self.ecg_filt)}')
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
        return df_output

    def plot_feature(self,df_Output):
        plt.close('all')
        fig, axs = plt.subplots(2, 1,sharex=True)
        fig.set_size_inches(15,9)

        axs[0].plot(df_Output['ecg_filt'])
        axs[0].scatter(x=df_Output.index, y=df_Output['T'], c='y', s=40, label='T')
        axs[0].scatter(x=df_Output.index, y=df_Output['P'], c='c', s=40, label='P')
        axs[0].scatter(x=df_Output.index, y=df_Output['R'], c='r', s=40, label='R')
        axs[0].scatter(x=df_Output.index, y=df_Output['Q'], c='k', s=40, label='Q')
        axs[0].scatter(x=df_Output.index, y=df_Output['S'], c='m', s=40, label='S')
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
        plt.savefig(f'{self.plt_Feat_path}/{self.patient}_feat.png')
        if self.showplot: plt.show()
        plt.close()

    def main(self):

        df_Output = self.Build_DataFrame()
        self.plot_feature(df_Output)
        # self.Med_df_Out = self.df_median(df_Output)
        # for column in df_Output.columns:
        #     print(column,len(df_Output[column].dropna()))
        interp_output = df_Output.interpolate(method='polynomial',order=1)
        self.Med_df_Out = interp_output.round(3)
        self.Med_df_Out.to_csv(f'{self.regr_path}/{self.patient}.csv')


if __name__=='__main__':
    file = '3150378_0008.csv'#'3904308_pat.csv' #'3400715_pat.csv'     3904396_pat
    PrePS = PreProcess_Sintec(file=file)   #3600490   (3602521 shekam dard)
    PrePS.main()
    breakpoint()

    df_err = pd.DataFrame(columns=['patient', 'error'])
    for n,file in enumerate(os.listdir('./Patients/')):
        
         if len(file.split('.')) > 1:
            if file.split('.')[1] == 'csv':
                patient = file.split('_')[0]
                path = './Patients/'
                print(f'Patient: {patient} - {n}\{len(os.listdir(path))}')
            try:
                PrePS = PreProcess_Sintec(file=file)   #3600490   (3602521 shekam dard)
                PrePS.main()
            except Exception as e:
                print(f"{patient} didn't complete")
                print(e)
                
                df_err.loc[len(df_err)] = {'patient':patient,'error':e}
                # with open('./feature_error.txt','a') as f:
                #     f.write(f'\n{patient}\n')
                #     f.write(f'{e}\n')
                #     f.write('------>>>><<<<<-------')
            # except Exception as e:
            #     print(f'----->>> {e}')

    df_err.to_csv('./Feat_Error.csv')

    print(len(os.listdir('./Dataset')))

