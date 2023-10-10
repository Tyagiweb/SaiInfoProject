import pandas as pd
import mplfinance as mpf
import numpy as np
import pandas_ta as ta
import os
import pickle

#------------------------------------------------------------------------------------------------------------------------------
charts_list=("Candlestick","OHLC","Line","Renko","Point and Figure")

indicators_list=("RSI","MFI","VWAP","MOVING_AVERAGE","EXPONENTIAL_MOVING_AVERAGE",
                 "MOMENTUM",'ALPHA_TREND','ICHIMOKU_CLOUD','DONCHIAN_CHANNEL','MACD','SUPER_TREND')

panel_1=["RSI","MFI","MOMENTUM",'MACD']

panel_0=["MOVING_AVERAGE","EXPONENTIAL_MOVING_AVERAGE","VWAP",'ALPHA_TREND','ICHIMOKU_CLOUD','DONCHIAN_CHANNEL','SUPER_TREND']

indicator_without_param=["VWAP"]

modified_indicators=['ALPHA_TREND','ICHIMOKU_CLOUD','DONCHIAN_CHANNEL','MACD','SUPER_TREND']

#---------------------------------------------------------------------------------------------------------------------

class indicator_class:
        def __init__(self, df,ct=None):
            self.df = df
            self.ct=ct
                                                                                   
        def moving_average(self,extra_params):
            #print(extra_params,'extprms')
            n=int(extra_params['n'])
            return self.df['Close'].rolling(window=n).mean(n)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        def exponential_moving_average(self, extra_params):
            n = int(extra_params['n']) 
            ema=self.df.ewm(span=n, adjust=False).mean()   
            #print(ema,'ema')                                                                                     
            return ema
                               
        def momentum(self,extra_params):
            n=int(extra_params['period'])               
            return  self.df['Close'] - self.df['Close'].shift(n)

        def rsi(self,extra_params):
            #print(extra_params)
            rsi_period=int(extra_params['period'])  
            #print(rsi_period,'rsi_period')                                                                                                            
            close_prices = self.df['Close']
            #print(close_prices.shape)
            price_diff = close_prices.diff()
            gain = price_diff.where(price_diff > 0, 0)
            loss = -price_diff.where(price_diff < 0, 0)
            #print(gain.shape,loss.shape)
            avg_gain = gain.rolling(window=rsi_period).mean()
            avg_loss = loss.rolling(window=rsi_period).mean()
            relative_strength = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + relative_strength))
            #print(rsi,'rsi_func')
            return rsi
        
        def mfi(self,extra_params):
            #print(extra_params)
            mfi_period=int(extra_params['period'])
            typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
            money_flow = typical_price * self.df['Volume']
            positive_money_flow = money_flow * (typical_price > typical_price.shift(1))
            negative_money_flow = money_flow * (typical_price < typical_price.shift(1))
            positive_money_flow_sum = positive_money_flow.rolling(window=mfi_period).sum()
            negative_money_flow_sum = negative_money_flow.rolling(window=mfi_period).sum()
            money_ratio = positive_money_flow_sum / negative_money_flow_sum
            mfi = 100 - (100 / (1 + money_ratio))
            return mfi
        
        def vwap(self):
            typical_prices = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
            return (typical_prices * self.df['Volume']).cumsum() / self.df['Volume'].cumsum()
        
        def alpha_trend(self,extra_params,ct=None): 
             
            Open = self.df['Open']
            Close = self.df['Close']
            High = self.df['High']
            Low = self.df['Low']
            Volume = self.df['Volume']
            ap=int(extra_params['ap'])
            tr = ta.true_range(High, Low, Close)
            atr = ta.sma(tr, ap)
            noVolumeData = False
            coeff = 1
            upt = [] 
            downT = []
            AlphaTrend = [0.0]
            src = Close
            rsi = ta.rsi(src, 14)
            hlc3 = []
            k1 = []
            k2 = []
            mfi = ta.mfi(High, Low, Close, Volume, 14)
            for i in range(len(Close)):
                hlc3.append((High[i] + Low[i] + Close[i]) / 3)

            for i in range(len(Low)):
                if pd.isna(atr[i]):
                    upt.append(0)
                else:
                    upt.append(Low[i] - (atr[i] * coeff))
            for i in range(len(High)):
                if pd.isna(atr[i]):
                    downT.append(0)
                else:
                    downT.append(High[i] + (atr[i] * coeff))
            for i in range(1, len(Close)):
                if noVolumeData is True and rsi[i] >= 50:
                    if upt[i] < AlphaTrend[i - 1]:
                        AlphaTrend.append(AlphaTrend[i - 1])
                    else:
                        AlphaTrend.append(upt[i])

                elif noVolumeData is False and mfi[i] >= 50:
                    if upt[i] < AlphaTrend[i - 1]:
                        AlphaTrend.append(AlphaTrend[i - 1])
                    else:
                        AlphaTrend.append(upt[i])
                else:
                    if downT[i] > AlphaTrend[i - 1]:
                        AlphaTrend.append(AlphaTrend[i - 1])
                    else:
                        AlphaTrend.append(downT[i])

            for i in range(len(AlphaTrend)):
                if i < 2:
                    k2.append(0)
                    k1.append(AlphaTrend[i])
                else:
                    k2.append(AlphaTrend[i - 2])
                    k1.append(AlphaTrend[i])

            #print(ct,ct+1)
            self.df[f'k{ct}'] = k1
            self.df[f'k{ct+1}'] = k2

            return self.df
        
        def ichimoku_cloud(self,extra_params,ct=None):
            periods = int(extra_params['periods'])
            self.df[f'Tenkan-sen{ct}'] = (self.df['High'].rolling(window=9).max() + self.df['Low'].rolling(window=9).min()) / 2
            self.df[f'Kijun-sen{ct}'] = (self.df['High'].rolling(window=26).max() + self.df['Low'].rolling(window=26).min()) / 2 
            self.df[f'Senkou_Span_A{ct}'] = (self.df[f'Tenkan-sen{ct}'] + self.df[f'Kijun-sen{ct}']) / 2 
            self.df[f'Senkou_Span_B{ct}'] = (self.df['High'].rolling(window=52).max() + self.df['Low'].rolling(window=52).min()) / 2 
            self.df[f'Chikou_Span{ct}'] = self.df['Close'].shift(periods=-periods) 
            return self.df

        def donchian_channel(self,extra_params,ct=None):
            period=int(extra_params['period'])
            self.df[f'Upper{ct}'] = self.df['High'].rolling(period).max()
            self.df[f'Lower{ct}'] = self.df['Low'].rolling(period).min()
            self.df[f'Middle{ct}'] = (self.df[f'Upper{ct}'] + self.df[f'Lower{ct}']) / 2
            return self.df
        
        def macd(self,extra_params,ct=None):
            macd_s_min_periods = int(extra_params['macd_s_min_periods'])
            #Get the 26-day EMA of the closing price
            k = self.df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()

            #Get the 12-day EMA of the closing price
            d = self.df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()

            #Subtract the 26-day EMA from the 12-Day EMA to get the MACD
            macd = k - d

            #Get the 9-Day EMA of the MACD for the Trigger line
            macd_s = macd.ewm(span=9, adjust=False, min_periods=macd_s_min_periods).mean()

            #Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
            macd_h = macd - macd_s
            #Add all of our new values for the MACD to the dataframe
            self.df[f'MACD{ct}'] = self.df.index.map(macd)
            self.df[f'MACDh{ct}'] = self.df.index.map(macd_h)
            self.df[f'MACDs{ct}'] = self.df.index.map(macd_s)
            return self.df
        
        def super_trend(self,extra_params,ct=None):
                    length = int(extra_params['period'])
                    multiplier=int(extra_params['ATR'])
                    st = ta.supertrend(self.df['High'], self.df['Low'], self.df['Close'], length=length, multiplier=multiplier,append=True)
                    self.df = self.df.join(st)
                    UP = []
                    DOWN = []
                    #print(self.df.columns,'supertrend_columns===========================================>>>>>>>>>>>>>>>>>>>>>>>')
                    for i in range(len(self.df)):
                        if self.df[f'SUPERTl_{length}_{float(multiplier)}'][i] < self.df['Close'][i]:
                            UP.append(float(self.df[f'SUPERTl_{length}_{float(multiplier)}'][i]))
                            DOWN.append(np.nan)
                        elif self.df[f'SUPERTs_{length}_{float(multiplier)}'][i] > self.df['Close'][i]:
                            DOWN.append(float(self.df[f'SUPERTs_{length}_{float(multiplier)}'][i]))
                            UP.append(np.nan)
                        else:
                            UP.append(np.nan)
                            DOWN.append(np.nan)
                    self.df[f'up{ct}'] = UP                           
                    self.df[f'down{ct}'] = DOWN                                                                                                                 
                    return self.df
                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
class indicators_:
        def __init__(self):
                fp=os.path.join(os.getcwd(),"mpl_canvas.pkl")
                with open(fp, 'rb') as file:
                    self.data=pickle.load(file)
                    for keys,values in self.data.items():
                        setattr(self, keys, values)
                self.flag=False

        def indicator_list_func(self):
                i=self.data['indicator_type']
                #print(i,'i')
                if i: 
                      
                        #print(i,'in')
                        counter=self.data[f'{i}_counter']
                        ct=self.data[f'{i}_cnt']
                        #print(ct,'ct')
                        var=None 
                        if i not in modified_indicators:
                            custom_params = {
                                'ylabel': i,
                                'secondary_y': True,
                                'ylim': (0, 100),
                                'color': '#00ff00',
                                'width': 1.5,
                                'y_on_right': True,
                                'linestyle': 'dashed',
                                'alpha': 0.7,
                                'marker': 'o',
                                #'markeredgecolor': 'red',
                                'markersize': 5,
                                #'fill': True,
                                #'fillcolor': 'lightblue'
                            }
                        
                        ind_params={}
                        #print(ct)
                        
                        self.data["extra_params"][f'{i}_{counter}']=self.indicators_parameters()
                        self.data['ind_ct'][f'{i}_{counter}_ct']=ct
                        #print(self.data["extra_params"])
                  
                        #print(self.data["extra_params"][f'{i}_{counter}'],'selfdatacounter')
                            
                        #print(self.data['indicators_dict'])
                        #print(self.data["extra_params"],"ind,ext_parm")
                                                                           
                        nested_instance = indicator_class(self.df)
                        #print(nested_instance)
                        if i  in indicator_without_param:
                            result = getattr(nested_instance, i.lower())()
                        else:
                            if i in modified_indicators:
                                
                                #ct=getattr(nested_instance(self.df), f'{i}_cnt')
                                result = getattr(nested_instance, i.lower())(self.data["extra_params"][f'{i}_{counter}'],ct=ct)
                                                          
                            else:
                                result = getattr(nested_instance, i.lower())(self.data["extra_params"][f'{i}_{counter}'])
                        if i in panel_0:
                            
                            if i in modified_indicators:
                               # print(f'outside- {i}')
                                                                                                   
                                if (i.lower()=='alpha_trend'):
                                    #print(f'inside {i}')
                                    alphatrend = result
                                    k1 = alphatrend[[f'k{ct}']].iloc[-self.num_bars:]
                                    #print(counter+ct+1)
                                    k2 = alphatrend[[f'k{ct+1}']].iloc[-self.num_bars:]
                                    #print(k1,k2)
                                    fill_up_color=self.data['extra_params'][f'{i}_{counter}']['fill_up_color']
                                    fill_down_color=self.data['extra_params'][f'{i}_{counter}']['fill_down_color']
                                    k1_color=self.data['extra_params'][f'{i}_{counter}']['k1_color']
                                    k2_color=self.data['extra_params'][f'{i}_{counter}']['k2_color']
                                    fill_up = dict(y1 = alphatrend[f'k{ct}'].values[-self.num_bars:], y2 = alphatrend[f'k{ct+1}'].values[-self.num_bars:], where = alphatrend[f'k{ct}'][-self.num_bars:] >= alphatrend[f'k{ct+1}'][-self.num_bars:], color = fill_up_color)
                                    fill_down = dict(y1 = alphatrend[f'k{ct}'].values[-self.num_bars:], y2 = alphatrend[f'k{ct+1}'].values[-self.num_bars:], where = alphatrend[f'k{ct}'][-self.num_bars:] <= alphatrend[f'k{ct+1}'][-self.num_bars:], color = fill_down_color)
                                    var = [mpf.make_addplot(k1,color = k1_color,width=3, secondary_y=False),mpf.make_addplot(k2,color = k2_color,width=3, secondary_y=False)]
                                    #var=mpf.plot(self.df.tail(90),addplot=ic,fill_between = [fill_up,fill_down])
                                    ind_params['fill_between']=[fill_up,fill_down]
                                    new_value = ct + 2
                                    setattr(self, f'{i}_cnt', new_value)
                                                                                          
                                if (i.lower()=='ichimoku_cloud'):
                                    ichimokucloud=result                                                                                                                                                                                                                                                                                         
                                    ich_cl1=ichimokucloud[[f'Tenkan-sen{ct}']].iloc[-self.num_bars:]
                                    ich_cl2=ichimokucloud[[f'Kijun-sen{ct}']].iloc[-self.num_bars:]
                                    ich_cl3=ichimokucloud[f'Chikou_Span{ct}'].iloc[-self.num_bars:]
                                    ich_cl4=ichimokucloud[[f'Senkou_Span_A{ct}']].iloc[-self.num_bars:]
                                    ich_cl5=ichimokucloud[[f'Senkou_Span_B{ct}']].iloc[-self.num_bars:]
                                    
                                    ich1_color=self.data['extra_params'][f'{i}_{counter}']['ich1_color']
                                    ich2_color=self.data['extra_params'][f'{i}_{counter}']['ich2_color']
                                    ich3_color=self.data['extra_params'][f'{i}_{counter}']['ich3_color']
                                    ich4_color=self.data['extra_params'][f'{i}_{counter}']['ich4_color']
                                    ich5_color=self.data['extra_params'][f'{i}_{counter}']['ich5_color']
                                    
                                    var= [
                                            mpf.make_addplot(ich_cl1,color=ich1_color,alpha=0.5, secondary_y=False),
                                            mpf.make_addplot(ich_cl2,color=ich2_color,alpha=0.5, secondary_y=False),
                                            mpf.make_addplot(ich_cl3,color=ich3_color,alpha=0.8, secondary_y=False),
                                            mpf.make_addplot(ich_cl4,color=ich4_color,alpha=0.8, secondary_y=False),
                                            mpf.make_addplot(ich_cl5,color=ich5_color,alpha=0.8, secondary_y=False)
                                        ]
                                    
                                    ich_fill_up_color=self.data['extra_params'][f'{i}_{counter}']['ich_fill_up_color']
                                    ich_fill_down_color=self.data['extra_params'][f'{i}_{counter}']['ich_fill_down_color']
                                    
                                    ichimoko_fill_up = dict(y1 = ichimokucloud[f'Senkou_Span_A{ct}'].values[-self.num_bars:], y2 = ichimokucloud[f'Senkou_Span_B{ct}'].values[-self.num_bars:], where = ichimokucloud[f'Senkou_Span_A{ct}'][-self.num_bars:] >= ichimokucloud[f'Senkou_Span_B{ct}'][-self.num_bars:], alpha = 0.5, color = ich_fill_up_color)
                                    ichimoko_fill_down = dict(y1 = ichimokucloud[f'Senkou_Span_A{ct}'].values[-self.num_bars:], y2 = ichimokucloud[f'Senkou_Span_B{ct}'].values[-self.num_bars:], where = ichimokucloud[f'Senkou_Span_A{ct}'][-self.num_bars:] < ichimokucloud[f'Senkou_Span_B{ct}'][-self.num_bars:], alpha =0.5,color=ich_fill_down_color)
                                    ind_params['fill_between']=[ichimoko_fill_up,ichimoko_fill_down]
                                    
                                    new_value = ct + 1
                                    setattr(self, f'{i}_cnt', new_value) 
                                    
                                if (i.lower()=='donchian_channel'):
                                    donchianchannel=result    
                                    dn_ch1_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch1_color']
                                    dn_ch2_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch2_color']
                                    dn_ch3_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch3_color']
                                    dn_chn_fill_color=self.data['extra_params'][f'{i}_{counter}']['dn_chn_fill_color']
                                    dn_ch1=donchianchannel[[f'Upper{ct}']].iloc[-self.num_bars:]   
                                    dn_ch2=donchianchannel[[f'Lower{ct}']].iloc[-self.num_bars:]  
                                    dn_ch3=donchianchannel[[f'Middle{ct}']].iloc[-self.num_bars:]     
                                    
                                    var=[
                                            mpf.make_addplot(dn_ch1,color=dn_ch1_color, secondary_y=False),
                                            mpf.make_addplot(dn_ch2,color=dn_ch2_color, secondary_y=False),
                                            mpf.make_addplot(dn_ch3,color=dn_ch3_color, secondary_y=False)
                                        ]
                                    
                                    donchian_fill = dict(y1=donchianchannel[f'Upper{ct}'].values[-self.num_bars:], y2=donchianchannel[f'Lower{ct}'].values[-self.num_bars:], alpha=0.1, color=dn_chn_fill_color)

                                    ind_params['fill_between']=[donchian_fill]
                                    new_value = ct + 1
                                    setattr(self, f'{i}_cnt', new_value) 
                                    
                                if (i.lower()=='super_trend'):

                                    supertrend=result
                                    
                                    up_super_trend = supertrend[[f'up{ct}']].iloc[-self.num_bars:]
                                    up_super_trend_color=self.data['extra_params'][f'{i}_{counter}']['up_super_trend_color']
                                    down_super_trend_color=self.data['extra_params'][f'{i}_{counter}']['down_super_trend_color']
                                    down_super_trend = supertrend[[f'down{ct}']].iloc[-self.num_bars:]
                                    var =  [
                                                #Supertrend
                                                mpf.make_addplot(up_super_trend,color = up_super_trend_color ,secondary_y=False),
                                                mpf.make_addplot(down_super_trend,color = down_super_trend_color , secondary_y=False),
                                                
                                            ]
                                    
                                    #Fill Between Method Appled
                                    fill_between_up_color=self.data['extra_params'][f'{i}_{counter}']['fill_between_up_color']
                                    fill_between_down_color=self.data['extra_params'][f'{i}_{counter}']['fill_between_down_color']
                                    fill_between_up=dict(y1=supertrend[f'up{ct}'].values[-self.num_bars:],y2=supertrend['Low'].values[-self.num_bars:],alpha=0.05,color=fill_between_up_color)
                                    fill_between_down=dict(y1=supertrend[f'down{ct}'].values[-self.num_bars:],y2=supertrend['High'].values[-self.num_bars:],alpha=0.05,color=fill_between_down_color)
                                    
                                    #print('================================>',fill_between_down,fill_between_up)
                                    #print('sp')
                                    ind_params['fill_between']=[fill_between_up,fill_between_down]     
                                    new_value = ct + 1
                                    setattr(self, f'{i}_cnt', new_value)
                                    #print(self.f'{i}_cnt')
                                                                                                                   
                            else:
                                var=mpf.make_addplot(result.iloc[-self.num_bars:],panel=0, ylabel=i, secondary_y=False,color='#00ff00')
                        else:   
                                if i in modified_indicators:
                                    if (i.lower()=='macd'):
                                        #Generating Colors For Histogram
                                        def gen_macd_color(df,ct):
                                            macd_color = []
                                            macd_color.clear()
                                            for i in range (0,len(df[f"MACDh{ct}"])):
                                                if df[f"MACDh{ct}"][i] >= 0 and df[f"MACDh{ct}"][i-1] < df[f"MACDh{ct}"][i]:
                                                    macd_color.append('#26A69A')
                                                    #print(i,'green')
                                                elif df[f"MACDh{ct}"][i] >= 0 and df[f"MACDh{ct}"][i-1] > df[f"MACDh{ct}"][i]:
                                                    macd_color.append('#B2DFDB')
                                                    #print(i,'faint green')
                                                elif df[f"MACDh{ct}"][i] < 0 and df[f"MACDh{ct}"][i-1] > df[f"MACDh{ct}"][i] :
                                                    #print(i,'red')
                                                    macd_color.append('#FF5252')
                                                elif df[f"MACDh{ct}"][i] < 0 and df[f"MACDh{ct}"][i-1] < df[f"MACDh{ct}"][i] :
                                                    #print(i,'faint red')
                                                    macd_color.append('#FFCDD2')
                                                else:
                                                    macd_color.append('#000000')
                                                    #print(i,'no')
                                            return macd_color
                                        
                                        macd_color = gen_macd_color(self.df,ct) 
                                        '''if self.data['extra_params'][f'{i}_{counter}']['macd_color']:
                                            macd_color=self.data['extra_params'][f'{i}_{counter}']['macd_color']'''
                                        
                                        maccd = result[[f'MACD{ct}']].iloc[-self.num_bars:]
                                        histogram = result[[f'MACDh{ct}']].iloc[-self.num_bars:]
                                        signal = result[[f'MACDs{ct}']].iloc[-self.num_bars:]  
                                        
                                        maccd_color=self.data['extra_params'][f'{i}_{counter}']['maccd_color']
                                        signal_color=self.data['extra_params'][f'{i}_{counter}']['signal_color']
                                        
                                        var = [
                                                mpf.make_addplot(maccd,color=maccd_color, panel=self.count+1),
                                                mpf.make_addplot(signal,color=signal_color, panel=self.count+1),
                                                mpf.make_addplot(histogram,type='bar',width=0.7,panel=self.count+1, color=macd_color,alpha=1,secondary_y=True),
                                            ]  
                                         
                                        new_value = ct + 1
                                        setattr(self, f'{i}_cnt', new_value)
                                   
                                else:
                                    var=mpf.make_addplot(result.iloc[-self.num_bars:], panel=self.count+1, **custom_params)
                                
                        if f'{i}_{counter}' not in  self.indicators_dict:
                            #self.current_indicator=f'{i}_{counter}'
                            self.indicators_dict[f'{i}_{counter}']=var
                            self.indicators_dict_params[f'{i}_{counter}']=ind_params
                            #print(self.data['indicators_dict'][f'{i}_{counter}'])
                            setattr(self, f"{i}_counter", counter+1)
                            if i in panel_1:
                              self.count+=1
                                                                                                                                                                                                                                                       
                self.dump_json_mpl_canvas()
                                                                                                                                                                                                                                                           
        def revised_indicator_edit(self,dt,indicator_type,pnl_count):
                
                #print(dt['extra_params'])
                self.data=dt
                #print(self.data['extra_params'])
                splt=indicator_type.rsplit("_",1)
                i=splt[0]   
                #print(i)                
                                                        
                if i:   
                        #self.data['indicator_type']=splt[0]
                        #self.data[f'{i}_counter']=splt[1]
                        #print(i,'in')
                        counter=splt[1]
                        ct=self.data['ind_ct'][f'{i}_{counter}_ct']
                        var=None 
                        if i not in modified_indicators:
                            custom_params = {
                                'ylabel': i,
                                'secondary_y': True,
                                'ylim': (0, 100),
                                'color': self.data['extra_params'][f'{i}_{counter}']['color'],
                                'width': 1.5,
                                'y_on_right': True,
                                'linestyle': 'dashed',
                                'alpha': 0.7,
                                'marker': 'o',
                                #'markeredgecolor': 'red',
                                'markersize': 5,
                                #'fill': True,
                                #'fillcolor': 'lightblue'
                            }
                        
                        ind_params={}
                        #print(ct)
                        #self.data["extra_params"][f'{i}_{counter}']=self.indicators_parameters()
                        #print(self.data["extra_params"][f'{i}_{counter}'],'selfdatacounter')
                                                                                                                                                                               
                        #print(self.data['indicators_dict'])
                        #print(self.data["extra_params"])
                                                                                                                                                                     
                        nested_instance = indicator_class(self.df)
                        #print(nested_instance)                        
                        if i  in indicator_without_param:                                    
                            result = getattr(nested_instance, i.lower())()
                        else:                                           
                            if i in modified_indicators:
                                                                                                   
                                #ct=getattr(nested_instance(self.df), f'{i}_cnt')
                                result = getattr(nested_instance, i.lower())(self.data["extra_params"][f'{i}_{counter}'],ct=ct)
                                                          
                            else:
                                result = getattr(nested_instance, i.lower())(self.data["extra_params"][f'{i}_{counter}'])
                        if i in panel_0:
                            
                            if i in modified_indicators:
                               # print(f'outside- {i}')
                                                                                                   
                                if (i.lower()=='alpha_trend'):
                                    #print(f'inside {i}')
                                    alphatrend = result
                                    k1 = alphatrend[[f'k{ct}']].iloc[-self.num_bars:]
                                    #print(counter+ct+1)
                                    k2 = alphatrend[[f'k{ct+1}']].iloc[-self.num_bars:]
                                    #print(k1,k2)
                                    fill_up_color=self.data['extra_params'][f'{i}_{counter}']['fill_up_color']
                                    fill_down_color=self.data['extra_params'][f'{i}_{counter}']['fill_down_color']
                                    k1_color=self.data['extra_params'][f'{i}_{counter}']['k1_color']
                                    k2_color=self.data['extra_params'][f'{i}_{counter}']['k2_color']
                                    fill_up = dict(y1 = alphatrend[f'k{ct}'].values[-self.num_bars:], y2 = alphatrend[f'k{ct+1}'].values[-self.num_bars:], where = alphatrend[f'k{ct}'][-self.num_bars:] >= alphatrend[f'k{ct+1}'][-self.num_bars:], color = fill_up_color)
                                    fill_down = dict(y1 = alphatrend[f'k{ct}'].values[-self.num_bars:], y2 = alphatrend[f'k{ct+1}'].values[-self.num_bars:], where = alphatrend[f'k{ct}'][-self.num_bars:] <= alphatrend[f'k{ct+1}'][-self.num_bars:], color = fill_down_color)
                                    var = [mpf.make_addplot(k1,color = k1_color,width=3, secondary_y=False),mpf.make_addplot(k2,color = k2_color,width=3, secondary_y=False)]
                                    #var=mpf.plot(self.df.tail(90),addplot=ic,fill_between = [fill_up,fill_down])
                                    ind_params['fill_between']=[fill_up,fill_down]
                                                                                          
                                if (i.lower()=='ichimoku_cloud'):
                                    ichimokucloud=result                                                                                                                                                                                                                                                                                         
                                    ich_cl1=ichimokucloud[[f'Tenkan-sen{ct}']].iloc[-self.num_bars:]
                                    ich_cl2=ichimokucloud[[f'Kijun-sen{ct}']].iloc[-self.num_bars:]
                                    ich_cl3=ichimokucloud[f'Chikou_Span{ct}'].iloc[-self.num_bars:]
                                    ich_cl4=ichimokucloud[[f'Senkou_Span_A{ct}']].iloc[-self.num_bars:]
                                    ich_cl5=ichimokucloud[[f'Senkou_Span_B{ct}']].iloc[-self.num_bars:]
                                    
                                    ich1_color=self.data['extra_params'][f'{i}_{counter}']['ich1_color']
                                    ich2_color=self.data['extra_params'][f'{i}_{counter}']['ich2_color']
                                    ich3_color=self.data['extra_params'][f'{i}_{counter}']['ich3_color']
                                    ich4_color=self.data['extra_params'][f'{i}_{counter}']['ich4_color']
                                    ich5_color=self.data['extra_params'][f'{i}_{counter}']['ich5_color']
                                    
                                    var= [
                                            mpf.make_addplot(ich_cl1,color=ich1_color,alpha=0.5, secondary_y=False),
                                            mpf.make_addplot(ich_cl2,color=ich2_color,alpha=0.5, secondary_y=False),
                                            mpf.make_addplot(ich_cl3,color=ich3_color,alpha=0.8, secondary_y=False),
                                            mpf.make_addplot(ich_cl4,color=ich4_color,alpha=0.8, secondary_y=False),
                                            mpf.make_addplot(ich_cl5,color=ich5_color,alpha=0.8, secondary_y=False)
                                        ]
                                    
                                    ich_fill_up_color=self.data['extra_params'][f'{i}_{counter}']['ich_fill_up_color']
                                    ich_fill_down_color=self.data['extra_params'][f'{i}_{counter}']['ich_fill_down_color']
                                    
                                    ichimoko_fill_up = dict(y1 = ichimokucloud[f'Senkou_Span_A{ct}'].values[-self.num_bars:], y2 = ichimokucloud[f'Senkou_Span_B{ct}'].values[-self.num_bars:], where = ichimokucloud[f'Senkou_Span_A{ct}'][-self.num_bars:] >= ichimokucloud[f'Senkou_Span_B{ct}'][-self.num_bars:], alpha = 0.5, color = ich_fill_up_color)
                                    ichimoko_fill_down = dict(y1 = ichimokucloud[f'Senkou_Span_A{ct}'].values[-self.num_bars:], y2 = ichimokucloud[f'Senkou_Span_B{ct}'].values[-self.num_bars:], where = ichimokucloud[f'Senkou_Span_A{ct}'][-self.num_bars:] < ichimokucloud[f'Senkou_Span_B{ct}'][-self.num_bars:], alpha =0.5,color=ich_fill_down_color)
                                    ind_params['fill_between']=[ichimoko_fill_up,ichimoko_fill_down]
                                    
                                if (i.lower()=='donchian_channel'):
                                    donchianchannel=result    
                                    dn_ch1_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch1_color']
                                    dn_ch2_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch2_color']
                                    dn_ch3_color=self.data['extra_params'][f'{i}_{counter}']['dn_ch3_color']
                                    dn_chn_fill_color=self.data['extra_params'][f'{i}_{counter}']['dn_chn_fill_color']
                                    dn_ch1=donchianchannel[[f'Upper{ct}']].iloc[-self.num_bars:]   
                                    dn_ch2=donchianchannel[[f'Lower{ct}']].iloc[-self.num_bars:]  
                                    dn_ch3=donchianchannel[[f'Middle{ct}']].iloc[-self.num_bars:]     
                                    
                                    var=[
                                            mpf.make_addplot(dn_ch1,color=dn_ch1_color, secondary_y=False),
                                            mpf.make_addplot(dn_ch2,color=dn_ch2_color, secondary_y=False),
                                            mpf.make_addplot(dn_ch3,color=dn_ch3_color, secondary_y=False)
                                        ]
                                    
                                    donchian_fill = dict(y1=donchianchannel[f'Upper{ct}'].values[-self.num_bars:], y2=donchianchannel[f'Lower{ct}'].values[-self.num_bars:], alpha=0.1, color=dn_chn_fill_color)

                                    ind_params['fill_between']=[donchian_fill]
                                    
                                if (i.lower()=='super_trend'):

                                    supertrend=result
                                    
                                    up_super_trend = supertrend[[f'up{ct}']].iloc[-self.num_bars:]
                                    up_super_trend_color=self.data['extra_params'][f'{i}_{counter}']['up_super_trend_color']
                                    down_super_trend_color=self.data['extra_params'][f'{i}_{counter}']['down_super_trend_color']
                                    down_super_trend = supertrend[[f'down{ct}']].iloc[-self.num_bars:]
                                    var =  [
                                                #Supertrend
                                                mpf.make_addplot(up_super_trend,color = up_super_trend_color ,secondary_y=False),
                                                mpf.make_addplot(down_super_trend,color = down_super_trend_color , secondary_y=False),
                                                
                                            ]
                                    
                                    #Fill Between Method Appled
                                    fill_between_up_color=self.data['extra_params'][f'{i}_{counter}']['fill_between_up_color']
                                    fill_between_down_color=self.data['extra_params'][f'{i}_{counter}']['fill_between_down_color']
                                    fill_between_up=dict(y1=supertrend[f'up{ct}'].values[-self.num_bars:],y2=supertrend['Low'].values[-self.num_bars:],alpha=0.05,color=fill_between_up_color)
                                    fill_between_down=dict(y1=supertrend[f'down{ct}'].values[-self.num_bars:],y2=supertrend['High'].values[-self.num_bars:],alpha=0.05,color=fill_between_down_color)
                                    
                                    #print('================================>',fill_between_down,fill_between_up)
                                    #print('sp')
                                    ind_params['fill_between']=[fill_between_up,fill_between_down]     
                                                                                                                   
                            else:
                                var=mpf.make_addplot(result.iloc[-self.num_bars:],panel=0, ylabel=i, secondary_y=False,color= self.data['extra_params'][f'{i}_{counter}']['color'])
                        else:   
                                if i in modified_indicators:
                                    if (i.lower()=='macd'):
                                        #Generating Colors For Histogram
                                        def gen_macd_color(df,ct):
                                            macd_color = []
                                            macd_color.clear()
                                            for i in range (0,len(df[f"MACDh{ct}"])):
                                                if df[f"MACDh{ct}"][i] >= 0 and df[f"MACDh{ct}"][i-1] < df[f"MACDh{ct}"][i]:
                                                    macd_color.append('#26A69A')
                                                    #print(i,'green')
                                                elif df[f"MACDh{ct}"][i] >= 0 and df[f"MACDh{ct}"][i-1] > df[f"MACDh{ct}"][i]:
                                                    macd_color.append('#B2DFDB')
                                                    #print(i,'faint green')
                                                elif df[f"MACDh{ct}"][i] < 0 and df[f"MACDh{ct}"][i-1] > df[f"MACDh{ct}"][i] :
                                                    #print(i,'red')
                                                    macd_color.append('#FF5252')
                                                elif df[f"MACDh{ct}"][i] < 0 and df[f"MACDh{ct}"][i-1] < df[f"MACDh{ct}"][i] :
                                                    #print(i,'faint red')
                                                    macd_color.append('#FFCDD2')
                                                else:
                                                    macd_color.append('#000000')
                                                    #print(i,'no')
                                            return macd_color
                                        
                                        macd_color = gen_macd_color(self.df,ct) 
                                        '''if self.data['extra_params'][f'{i}_{counter}']['macd_color']:
                                            macd_color=self.data['extra_params'][f'{i}_{counter}']['macd_color']'''
                                            
                                        maccd = result[[f'MACD{ct}']].iloc[-self.num_bars:]
                                        histogram = result[[f'MACDh{ct}']].iloc[-self.num_bars:]
                                        signal = result[[f'MACDs{ct}']].iloc[-self.num_bars:]  
                                        
                                        maccd_color=self.data['extra_params'][f'{i}_{counter}']['maccd_color']
                                        signal_color=self.data['extra_params'][f'{i}_{counter}']['signal_color']
                                         
                                        var = [
                                                mpf.make_addplot(maccd,color=maccd_color, panel=pnl_count),
                                                mpf.make_addplot(signal,color=signal_color, panel=pnl_count),
                                                mpf.make_addplot(histogram,type='bar',width=0.7,panel=pnl_count, color=macd_color,alpha=1,secondary_y=True),
                                            ]  
                                         
                                else:
                                    var=mpf.make_addplot(result.iloc[-self.num_bars:], panel=pnl_count, **custom_params)
                            
                        self.data['indicators_dict'][f'{i}_{counter}']=var
                        #print(self.data['indicators_dict'][f'{i}_{counter}'])
                        self.data['indicators_dict_params'][f'{i}_{counter}']=ind_params    
                
                fp=os.path.join(os.getcwd(),'mpl_canvas.pkl')
                with open(fp, 'wb') as file:
                   pickle.dump(self.data, file)
                        
        def dump_json_mpl_canvas(self):
                    data = {
                        "file_path": self.file_path,
                        "num_bars": self.num_bars,
                        "stock_name": self.stock_name,
                        "chart_type": self.chart_type,
                        "indicator_type": self.indicator_type,
                        "count": self.count,
                        #"panel_1_count": self.panel_1_count,
                        "indicators_dict": self.indicators_dict,
                        "indicators_dict_params": self.indicators_dict_params,
                        "df":self.df,
                        "extra_params":self.extra_params,
                        'ind_ct':self.ind_ct
                        
                    }
                    
                    for i in indicators_list:
                        data[f"{i}_counter"] = getattr(self, f"{i}_counter")
                        data[f"{i}_cnt"] = getattr(self, f"{i}_cnt")

                    fp=os.path.join(os.getcwd(),'mpl_canvas.pkl')
                    with open(fp, 'wb') as file:
                       pickle.dump(data, file) 
                                                            
        def indicators_parameters(self):
                    indicator_type=self.data['indicator_type']
                    # print(indicator_type,'indicators_parameters')
                    d=None
                    if indicator_type=='RSI':
                        d={'period':14,'color':'#00ff00'}
                      
                    if indicator_type=='MFI':
                        d={'period':14,'color':'#00ff00'}
                
                    if indicator_type=="MOMENTUM":
                        d={'period':14,'color':'#00ff00'}
               
                    if indicator_type=="MOVING_AVERAGE":

                        d={'n':5,'color':'#00ff00'}
                        
                    if indicator_type=="EXPONENTIAL_MOVING_AVERAGE":

                        d={'n':5,'color':'#00ff00'}
                        
                    if indicator_type=="VWAP":

                        d={'color':'#00ff00'}
                        
                    if indicator_type=="DONCHIAN_CHANNEL":

                        d={'period':14,'dn_ch1_color':'#2962FF','dn_ch2_color':'#FF6D00','dn_ch3_color':'#2962FF',
                           'dn_chn_fill_color':'#2962FF'}
                        
                    if indicator_type=="SUPER_TREND":

                        d={'period':7,'ATR':3,'up_super_trend_color':'green','down_super_trend_color':'#FF8849',
                           'fill_between_up_color':'#00FF00','fill_between_down_color':'#FF0000'
                           }
                        
                    if indicator_type=="ALPHA_TREND":

                        d={'ap':14,'fill_up_color':'r',"fill_down_color":'g',"k1_color":"#0022FC","k2_color":'#FC0400'}
                        
                    if indicator_type=="ICHIMOKU_CLOUD":

                        d={'color':'#00ff00'}
                        
                    if indicator_type=="ICHIMOKU_CLOUD":

                        d={'periods':26,"ich1_color":'#fcc905',"ich2_color":'#F83C78',"ich3_color":'#8D8D16',"ich4_color":'#006B3D',
                           "ich5_color":'#D3212C',"ich_fill_up_color":'#00FF00',"ich_fill_down_color":'#FF0000'}
                        
                    if indicator_type=="MACD":
                        d={'macd_s_min_periods':9,'maccd_color':'#2962FF','signal_color':'#FF6D00'}
                    
                    return d