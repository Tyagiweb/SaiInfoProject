import sys
import pandas as pd
import mplfinance as mpf
import os
import mplcursors
from matplotlib import pyplot as plt
from indicators_python02__1 import indicators_,indicators_list,charts_list,panel_1
import pickle
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import pyqtSignal,Qt,QSize,QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QMdiArea, QMdiSubWindow, QTextEdit, QAction, QMenu, QMenuBar, 
                             QToolBar, QFileDialog, QDialog, QTabWidget, QLabel, QGridLayout, QLineEdit, 
                             QPushButton, QCheckBox, QRadioButton, QButtonGroup, QComboBox, QSpinBox, QDoubleSpinBox, 
                             QDateEdit, QTimeEdit, QDateTimeEdit, QTextEdit, QWidget,QFrame,QSpacerItem,QDialogButtonBox,QSizePolicy,
                             QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem
                             )

class ListDisplayDialog(QDialog):
    def __init__(self, items_list, parent=None):
        super().__init__(parent)

        # Allow minimize and maximize
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)

        self.setWindowTitle("Delete Indicators List")
        self.setFixedSize(QSize(800, 600))
        
        mainLayout = QHBoxLayout(self)

        # Left Layout for the checkbox list
        leftLayout = QVBoxLayout()
        self.listWidget = QListWidget(self)

        # Populate the list widget with checkboxes
        for item_text in items_list:
            item = QListWidgetItem(self.listWidget)
            checkbox = QCheckBox(item_text)
            checkbox.setChecked(True)  # Set the checkbox to be initially checked
            item.setSizeHint(checkbox.sizeHint())  # Adjust size hint for proper display
            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item, checkbox)

        leftLayout.addWidget(self.listWidget)
        mainLayout.addLayout(leftLayout, 45)  # Use stretch factor here

        # Right Layout
        rightLayout = QVBoxLayout()
        # Add stretch to push the buttons to the bottom
        rightLayout.addStretch(1)

        # Buttons layout
        buttonsLayout = QHBoxLayout()

        okButton = QPushButton("OK", self)
        okButton.clicked.connect(self.accept)
        buttonsLayout.addWidget(okButton)

        closeButton = QPushButton("Close", self)
        closeButton.clicked.connect(self.close)
        buttonsLayout.addWidget(closeButton)

        rightLayout.addLayout(buttonsLayout)

        mainLayout.addLayout(rightLayout, 55)  # Use stretch factor here

        self.setLayout(mainLayout)
        
    def get_checked_items(self):
        checked_items = []
        for index in range(self.listWidget.count()):
            item = self.listWidget.item(index)
            checkbox = self.listWidget.itemWidget(item)
            if checkbox.isChecked():
                checked_items.append(checkbox.text())
        return checked_items
    
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QFrame, QPushButton, QScrollArea, QLabel, QLineEdit, QSizeGrip, QFormLayout)
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
from functools import partial

class ListDisplayDialog1(QDialog):

    def __init__(self, dt, parent=None):
        self.data=dt
        #print(dt)
        items_list=list(dt["indicators_dict"].keys())
        super().__init__(parent)

        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle("Indicators List")
        self.setFixedSize(QSize(800, 600))
        self.lineEdits = {}
        outerLayout = QVBoxLayout(self)

        mainLayout = QHBoxLayout()
        outerLayout.addLayout(mainLayout)

        # Left Layout (List box)
        self.listWidget = QListWidget(self)
        self.listWidget.addItems(items_list)
        mainLayout.addWidget(self.listWidget, 1)  # Make it occupy half the dialog width

        # Connect to item clicked signal
        self.listWidget.itemClicked.connect(self.on_item_clicked)

        # Right Layout (Scroll Area for parameters)
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        mainLayout.addWidget(self.scrollArea, 1)  # Same weight as left layout

        self.scrollAreaWidgetContents = QFrame(self.scrollArea)
        self.formLayout = QFormLayout(self.scrollAreaWidgetContents)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)

        # Buttons layout
        buttonsLayout = QHBoxLayout()
        outerLayout.addLayout(buttonsLayout)

        # Space filler to push buttons to the right
        buttonsLayout.addStretch()

        okButton = QPushButton("OK", self)
        okButton.clicked.connect(self.accept)
        buttonsLayout.addWidget(okButton)

        closeButton = QPushButton("Cancel", self)
        closeButton.clicked.connect(self.reject)
        buttonsLayout.addWidget(closeButton)

        self.setLayout(outerLayout)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def on_item_clicked(self, item):
        self.selected_item = item.text()

        # Clear the right box contents
        self.clear_layout(self.formLayout)
        self.lineEdits.clear()
            
    #-------------------------------------------------------------------------------------------------------

        parameters_mpf_addplot = list(self.data['extra_params'][self.selected_item].keys())

        for param in parameters_mpf_addplot:
            label = QLabel(param, self)

            if 'color' in param.lower():  # Check if the param contains the word 'color'
                colorButton = QPushButton(self)
                
                # Get the initial color
                initialColor = QColor(self.data['extra_params'][self.selected_item][param])
                
                # Set the background color of the button
                colorButton.setStyleSheet(f"background-color: {initialColor.name()};")
                
                # Connect to the function to change the color
                colorButton.clicked.connect(partial(self.selectColor, colorButton, param))
                
                self.formLayout.addRow(label, colorButton)
                self.lineEdits[param] = colorButton

            else:
                lineEdit = QLineEdit(self)
                lineEdit.setText(str(self.data['extra_params'][self.selected_item][param]))
                self.formLayout.addRow(label, lineEdit)
                self.lineEdits[param] = lineEdit

    def selectColor(self, button, paramName):
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")
            #button.setText(color.name())  # Update the button's text to the new color name
            self.data['extra_params'][self.selected_item][paramName] = color.name()
    
    #-------------------------------------------------------------------------------------------------------
            
    def get_selected_item(self):
        return self.selected_item

    def get_parameters_values(self):
        def indicator_panel_count(dt):
            for i,v in dt["indicators_dict"].items():
                if i==self.selected_item:
                    #print(len(v),'ln=v')                                                               
                    if i.rsplit('_',1)[0] in panel_1:
                            #print(i,len(v),"lenv")
                                                                                                                         
                            if len(v)==24:
                                    #print(i,'len==24')
                                    #print(v['panel'])
                                                                                                                
                                    return v['panel']  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                            else:  
                                                                                                      
                                for j in v:
                                       return j['panel']
        #print([le.text() if isinstance(le, QLineEdit) else le.palette().button().color().name() for param, le in self.lineEdits.items()])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        """Retrieve the parameters and their corresponding input values."""
        pr= {param: (le.text() if isinstance(le, QLineEdit) else le.palette().button().color().name()) for param, le in self.lineEdits.items()}
       # print(pr)
        self.data['extra_params'][self.selected_item]=pr
        dt=self.data
        pnl_count=indicator_panel_count(dt)
        #print(pnl_count,'pnl_count')
        indicators_().revised_indicator_edit(dt,self.selected_item,pnl_count)
        fp = os.path.join(os.getcwd(), "mpl_canvas.pkl")
        with open(fp, 'rb') as file:
                dt = pickle.load(file)
                
        return dt
                                         
    # [Accept and reject redefined to demonstrate retrieving parameter values]
    def accept(self):
        #print(self.get_parameters_values())  # You can modify this part to handle the retrieved values as required
        super().accept()
                                  
    def reject(self):
        super().reject()
                                                         
class DataLoader:
    def __init__(self, file_path='mm.txt', num_bars=100):
        self.file_path = file_path
        self.num_bars = num_bars
        self.column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.df = None
        self.load_data()
    
    def load_data(self):
        if self.file_path:
            self.df = pd.read_csv(self.file_path, header=None, names=self.column_names)
    
        # Convert time to a valid format
        self.df['Time'] = self.df['Time'].apply(lambda x: str(x).zfill(4) if len(str(x)) < 4 else str(x))
        self.df['Time'] = self.df['Time'].str[:2] + ':' + self.df['Time'].str[2:]
        
        # Now combine 'Date' and 'Time' columns
        self.df['Date'] = pd.to_datetime(self.df['Date'].astype(str) + ' ' + self.df['Time'].astype(str),dayfirst=True)
        
        # Drop the 'Time' column as it's no longer needed
        self.df.drop(columns='Time', inplace=True)
        
        # Set the 'Date' column as the index
        self.df.set_index('Date', inplace=True)
        
        self.df.drop_duplicates(inplace=True)
        
        self.c_df=self.df.iloc[-self.num_bars:]
        #br=3*self.num_bars
        self.df=self.df.iloc[-(3*self.num_bars)-self.num_bars:]
        
    def refresh_data(self):
        self.load_data()
        
class MplCanvas(FigureCanvas):
    
    def __init__(self, file_path, num_bars, parent=None):
        self.parent=parent
        self.flag=None
        self.file_path = file_path
        self.num_bars = num_bars
        self.stock_name=file_path.split('/')[-1].split('.')[0].replace('!','')
        self.fig = Figure()
        super().__init__(self.fig)
        lst=['line_drawer','cidpress','cidrelease','cidmove','cursor','ax','axes']
        for i in lst:
            setattr(self, i, None)
                                                                                    
        self.indicators_dict={}
        self.indicators_dict_params={}
                                                                                           
        self.chart_type = 'candle'  # Default chart type
        self.edit_type=None
        self.indicator_type=''
        self.count=0
        self.extra_params={}
        #self.panel_1_count=0
                                                         
        for i in indicators_list:
            setattr(self, f"{i}_counter", 0)
            setattr(self, f"{i}_cnt", 1)
                                                                     
        self.data_loader = DataLoader(file_path, num_bars)
        self.df = self.data_loader.df
        self.c_df = self.data_loader.c_df 
        self.reset_data=self.df
        self.ind_ct={}
        self.dump_json_mpl_canvas()
        self.update_chart()
        self.set_cursor()
        
        #print(self.indicators_dict,'self.indicatos_dict')
                                                       
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
                                                                            
    def set_cursor(self):
        # After you create the plot
        self.cursor = mplcursors.cursor(self.ax, hover=False)

        # Add this function to display the information you want when the cursor is clicked
        @self.cursor.connect("add")
        def on_add(sel):
            x, y = sel.target
            sel.annotation.set_text('Date: {}\nTime: {}\nOpen: {}\nHigh: {}\nLow: {}\nClose: {}\nVolume: {}'.format(
                self.df.index[int(x)].date(), self.df.index[int(x)].time(), self.df.iloc[int(x)]['Open'],
                self.df.iloc[int(x)]['High'], self.df.iloc[int(x)]['Low'], self.df.iloc[int(x)]['Close'],
                self.df.iloc[int(x)]['Volume']))
                                                                                                   
    def chng_pkl(self,fp):
            fp=os.path.join(os.getcwd(),"mpl_canvas.pkl")
            with open(fp, 'rb') as file:
                dt=pickle.load(file) 
            if dt :
                dt['indicator_type']=self.indicator_type
                dt['chart_type']=self.chart_type
                dt['file_path']=self.file_path
                dt['num_bars']=self.num_bars
                dt['stock_name']=self.stock_name
                dt["extra_params"]:self.extra_params

            with open(fp, 'wb') as file:
                       pickle.dump(dt, file) 
                                                                                                                                                                                  
    def update_chart(self):
        #print(self.indicator_type,'self.indicator_type_tst')                                                                                                            
        fp=os.path.join(os.getcwd(),"mpl_canvas.pkl")
        self.chng_pkl(fp)
        dt=None
                                                                                                                                                                                                                                                 
        if self.indicator_type or self.flag:
            indicators_().indicator_list_func()
            with open(fp, 'rb') as file:
                            dt=pickle.load(file)
                                                                                                                                                                                                                          
        exclude_chart_list=['renko','pnf'] 
                                                                                                
        #indicator_class(self.file_path,self.num_bars).indicator_list_func()
                                                                                                                               
        style = mpf.make_mpf_style(base_mpf_style='yahoo', gridcolor='none', gridstyle='')
           
        if self.chart_type in exclude_chart_list:
                plt.cla()
                fig, axes = mpf.figure(style=style), plt.gca()
                
                mpf.plot(self.df, type=self.chart_type, show_nontrading=False, ax=axes, axtitle='{}'.format(self.stock_name.capitalize()))
                
                #---------------------------------------------------------------------------------------------------------------------
                
                for v in self.indicators_dict.keys():
                    axes[0].plot(v['data'])
                
                #--------------------------------------------------------------------------------------------------------------------
        else:    
            ap=[]
            if dt:
                #print(dt["indicators_dict"])
                
                for i,v in dt["indicators_dict"].items():
                    if type(v)==list:
                        ap.extend(v)
                    else:
                        ap.append(v)   
            #print(ap)                 
                                                                             
            if dt and dt["indicators_dict_params"]: 
                #print(self.indicators_dict_params)
                fill_between=[]
                tr=['alpha_trend','ichimoku_cloud','donchian_channel','super_trend']
                for k,v in dt["indicators_dict_params"].items():
                    splt=k.lower().rsplit('_',1)
                    if splt[0] in tr:
                           #print(v)
                           fill_between.extend(v['fill_between'])
                                                                                                                                                                                                                            
                if fill_between:
                                                                                                                                                                  
                        fig, axes = mpf.plot(self.c_df ,style=style ,type=self.chart_type, show_nontrading=False, returnfig=True, addplot=ap ,fill_between = fill_between, axtitle='{}'.format(self.stock_name.split('.')[0].capitalize()),tight_layout=True)
                        
                else:
                        fig, axes = mpf.plot(self.c_df,style=style ,type=self.chart_type, show_nontrading=False, returnfig=True, addplot=ap, axtitle='{}'.format(self.stock_name.split('.')[0].capitalize()),tight_layout=True)
                                                                    
            else:
                if ap:
                    fig, axes = mpf.plot(self.c_df,style=style ,type=self.chart_type, show_nontrading=False, returnfig=True, addplot=ap, axtitle='{}'.format(self.stock_name.split('.')[0].capitalize()),tight_layout=True)
                else:                           
                    #print('line_else',dt,self.indicator_type)
                    fig, axes = mpf.plot(self.c_df,style=style ,type=self.chart_type, show_nontrading=False, returnfig=True,axtitle='{}'.format(self.stock_name.split('.')[0].capitalize()),tight_layout=True)
                                                                                                                                                                                                                                                   
        # assuming self is a FigureCanvasQTAgg
        self.figure =fig
        sb=self.parent.mdi_area.activeSubWindow()
        if sb and sb.isMaximized() and self.parent.isMaximized():
            fig.set_figwidth(20)  # Set the width of the figure in inches
            fig.set_figheight(9.5) 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        for idx,i in enumerate(axes):
            current_position=i.get_position()
                               
            new_position=[current_position.x0,current_position.y0,0.78,current_position.height-0.02]
            axes[idx].set_position(new_position)
        
        self.ax=axes
        self.draw()
        self.set_cursor()
                                                                                                                                           
    def change_chart_type(self, chart_type):
        self.chart_type = chart_type
        self.flag=True
        self.indicator_type=None
        self.update_chart()
                                                                                                                                           
    def change_indicator_type(self, indicator_type):
            self.indicator_type = indicator_type
            self.update_chart()
            
    def read_pckl(self):
        fp = os.path.join(os.getcwd(), "mpl_canvas.pkl")
        with open(fp, 'rb') as file:
                dt = pickle.load(file)
        return dt
        
    def write_pckl(self,dt):
        fp = os.path.join(os.getcwd(), "mpl_canvas.pkl")
        with open(fp, 'wb') as file:
            pickle.dump(dt, file) 
                                                                                                                                                                                                                                                                                                                                                                                                                                   
    def change_edit_type(self, edit_type):
        dt=self.read_pckl()
        self.edit_type=edit_type
        
        def indicator_panel_edit(dt,clk):
    
            clf=len([i for i in clk if i.split('_')[0] in panel_1])
            dt['count']=dt['count']-clf
            n=1
            for i,v in dt["indicators_dict"].items():

                    if i.rsplit('_',1)[0] in panel_1:
                        
                            if len(v)==24:
                                    
                                    v['panel']=n     
                                    #print(v['panel'],'if',i)
                                    n+=1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
                            else:  
   
                                for j in v:
                                    j['panel']=n
                                    #print(j['panel'],'else',i)
                                n+=1
                            #print()
                    #else:
                        #print('not in panel_1')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
            return dt                                                         
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
        if self.edit_type == 'EDIT':
            dt=self.read_pckl()
            dialog = ListDisplayDialog1(dt, self)
            #print(1)
            if dialog.exec_() == QDialog.Accepted:
                    checked_items = dialog.get_parameters_values()
                    dt=checked_items
                    #print(checked_items,'checked items')
                                                                                      
        if self.edit_type == 'DELETE':
            dialog = ListDisplayDialog(list(dt["indicators_dict"].keys()), self)
            #print(2)
            if dialog.exec_() == QDialog.Accepted:
                    checked_items = dialog.get_checked_items()
                    for i in checked_items:
                        if i in list(dt["indicators_dict"].keys()):
                            dt["indicators_dict"].pop(i)
                            dt["indicators_dict_params"].pop(i)
                    dt=indicator_panel_edit(dt,checked_items)
                                                                                           
        self.write_pckl(dt)                                                                             
        self.flag=True
        self.indicator_type=None
        self.update_chart()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
class ChartDialog(QDialog):
    def __init__(self, main_window):
        super().__init__(main_window)  # pass main_window here
        self.setWindowTitle("Open Chart")

        self.tab_widget = QTabWidget(self)
        self.tab1 = Tab1(main_window)  # pass main_window here
        self.tab2 = Tab2()

        self.tab_widget.addTab(self.tab1, "File")
        self.tab_widget.addTab(self.tab2, "Chart Details")

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(self.buttons)

        self.setLayout(main_layout)
        self.resize(400, 300)  # adjust the size of dialog


class Tab1(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout()

        self.file_open_button = QPushButton('Open File', self)
        self.file_open_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.file_open_button)

        self.bars_label = QLabel('Number of Bars:', self)
        layout.addWidget(self.bars_label)
 
        self.num_bars = QSpinBox(self)
        self.num_bars.setRange(1, 1000)  # Set the range of the spinbox
        self.num_bars.setValue(75)  # Set the default value
        layout.addWidget(self.num_bars)

        self.setLayout(layout)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = "/Trend Analyser/Data/intra5min/"
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "QFileDialog.getOpenFileName()", 
            directory, 
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
        options=options
        )
        if file_path:
            self.file_path = file_path  # Store file_path for later use


class Tab2(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        # Add any additional widgets you want in this tab
        
class MySubWindow(QMdiSubWindow):
    aboutToClose = pyqtSignal(QMdiSubWindow)  # Include the subwindow itself in the signal

    def __init__(self, widget):
        super().__init__()
        self.setWidget(widget)

    def closeEvent(self, event):
        self.aboutToClose.emit(self)  # Emit signal with self as argument
        super().closeEvent(event)

class LineDrawer:
    def __init__(self, ax):
        self.ax = ax
        self.line = None
        self.start_point = None
        self.end_point = None

    def mouse_press(self, event):
        if event.inaxes != self.ax:
            return

        if self.start_point is None:  # Start of a new line
           self.start_point = (event.xdata, event.ydata)
           self.line = self.ax.plot([self.start_point[0]], [self.start_point[1]], color='r')[0]
        else:  # End of the current line
            self.end_point = (event.xdata, event.ydata)
            self.line.set_data([self.start_point[0], self.end_point[0]], [self.start_point[1], self.end_point[1]])
            self.ax.figure.canvas.draw()

    def mouse_release(self, event):
        if event.inaxes != self.ax or self.start_point is None:
            return

        self.end_point = (event.xdata, event.ydata)
        self.line.set_data([self.start_point[0], self.end_point[0]], [self.start_point[1], self.end_point[1]])
        self.ax.figure.canvas.draw()

        # Prepare for a new line
        self.start_point = None
        self.line = None

    def mouse_move(self, event):
        if event.inaxes != self.ax or self.start_point is None or self.line is None:
            return
        
        # Update the end of the line to the current mouse position
        self.line.set_data([self.start_point[0], event.xdata], [self.start_point[1], event.ydata])
        self.ax.figure.canvas.draw()

    def draw_line(self):
        x_values = [self.start_point[0], self.end_point[0]]
        y_values = [self.start_point[1], self.end_point[1]]
        self.line = self.ax.plot(x_values, y_values, color='r')[0]

    def clear_line(self):
        if self.line:
            self.line.remove()
            self.line = None
        self.start_point = None
        self.end_point = None
    
class CustomToolbar(QToolBar):
    def __init__(self, mdi_window):
        super().__init__(mdi_window)

        # New actions
        instrument_action = QAction(QIcon("display-solid.png"), "Instrument", self)
        instrument_action.setToolTip("Instrument")
        instrument_action.triggered.connect(mdi_window.instrument_function)  # Placeholder function
        self.addAction(instrument_action)
        
        
        #-----------------------------------------------------------------------------------------
        edit_action = QAction(QIcon("EDIT_ICON.png"), "Indicators Edit", self)
        edit_action.setToolTip("Indicators Edit")
        edt=["EDIT","DELETE"]
        edit_types_menu = QMenu("Indicators_Edit", self)
        edit_action.setMenu(edit_types_menu )
        actions_edit = {}
        for i in edt:
            action_name = f"{i.upper()}"
            actions_edit[action_name] = QAction(f"{i}", self)
                                                                      
            actions_edit[action_name].triggered.connect(lambda _,i=i: mdi_window.change_edit_type(i))
            edit_types_menu.addAction(actions_edit[action_name])
                                                                                                                                                                                         
        self.addAction(edit_action)
        #---------------------------------------------------------------------------------------

        #---------------------------------------------------------------------
        indicator_action = QAction(QIcon("Keyboard-regular.png"), "Indicator Types", self)
        indicator_action.setToolTip("Indicator Types")

        indicator_types_menu = QMenu("Indicator Types", self)
        indicator_action.setMenu(indicator_types_menu)
        actions = {}
        for i in indicators_list:
            action_name = f"{i.upper()}"
            actions[action_name] = QAction(f"{i}", self)

            actions[action_name].triggered.connect(lambda _,i=i: mdi_window.change_indicator_type(i))
            indicator_types_menu.addAction(actions[action_name])

        self.addAction(indicator_action)

        # Add spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(spacer)
        spacer.setMaximumWidth(50)  # Adjust this value to your needs

        strategy_action = QAction(QIcon("Handshake-regular.png"), "Strategy", self)
        strategy_action.setToolTip("Strategy")
        strategy_action.triggered.connect(mdi_window.strategy_function)  # Placeholder function
        self.addAction(strategy_action)

        strategy_builder_action = QAction(QIcon("code-solid.png"), "Strategy Builder", self)
        strategy_builder_action.setToolTip("Strategy Builder")
        strategy_builder_action.triggered.connect(mdi_window.strategy_builder_function)  # Placeholder function
        self.addAction(strategy_builder_action)

        # Add separator
        self.addSeparator()

        grid_display_action = QAction(QIcon("table-cells-large-solid.png"), "Grid Display", self)
        grid_display_action.setToolTip("Grid Display")
        grid_display_action.triggered.connect(mdi_window.grid_display_function)  # Placeholder function
        self.addAction(grid_display_action)

        scanner_action = QAction(QIcon("squarespace.png"), "Scanner", self)
        scanner_action.setToolTip("Scanner")
        scanner_action.triggered.connect(mdi_window.scanner_function)  # Placeholder function
        self.addAction(scanner_action)

    # The existing actions
        open_chart_action = QAction(QIcon("folder-plus-solid.png"), "Open Chart", self)
        open_chart_action.setToolTip("Open Chart")
        open_chart_action.triggered.connect(mdi_window.open_chart)
        self.addAction(open_chart_action) 

        chart_types_action = QAction(QIcon("chart-line-solid.png"), "Chart Types", self)
        chart_types_action.setToolTip("Chart Types")

        chart_types_menu = QMenu("Chart Types", self)
        chart_types_action.setMenu(chart_types_menu)

        actions_chart = {}
        for i in charts_list:
            action_name = f"{i.upper()}"
            actions_chart[action_name] = QAction(f"{i}", self)
            if i=="Candlestick":
                actions_chart[action_name].triggered.connect(lambda _,i=i: mdi_window.change_chart_type('candle'))

            elif i=="Point and Figure":
                actions_chart[action_name].triggered.connect(lambda _,i=i: mdi_window.change_chart_type('pnf'))
                
            else:
                actions_chart[action_name].triggered.connect(lambda _,i=i: mdi_window.change_chart_type(i.lower()))
            chart_types_menu.addAction(actions_chart[action_name])

        self.addAction(chart_types_action)

        # refresh_chart_action = QAction(QIcon("mug-hot-solid.png"), "Refresh Chart", self)
        refresh_chart_action = QAction(QIcon("refersh-chart.png"), "Refresh Chart", self)
        refresh_chart_action.setToolTip("Refresh Chart")
        refresh_chart_action.triggered.connect(mdi_window.refresh_chart)
        self.addAction(refresh_chart_action)

        trendline_icon = QIcon("./pencil-solid.png")
        trendline_action = QAction(trendline_icon, "Draw Trendline", self)
        trendline_action.triggered.connect(mdi_window.toggle_trendline_mode)
        self.addAction(trendline_action)

    # Move help icon to the end
        help_action = QAction(QIcon("question-solid.png"), "Help", self)
        help_action.setToolTip("Help")
        self.addAction(help_action)      
                                                                                                                                                                                                                    
class MdiWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MDI Window Example")
        self.setGeometry(100, 100, 800, 650)

        self.canvas = None  # Add this       

        self.mdi_area = QMdiArea()
        self.setCentralWidget(self.mdi_area)

         # Initialize QLabel for displaying messages
        self.message_label = QLabel(self)
        # Position it somewhere visible
        self.message_label.setGeometry(100, 100, 200, 20)

        # Initialize custom toolbar
        self.toolbar = CustomToolbar(self)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)

        menu_bar = self.menuBar()

        file_menu = QMenu("File", self)
        menu_bar.addMenu(file_menu)

        window_menu = QMenu("Window", self)
        menu_bar.addMenu(window_menu)

        new_window_action = QAction("New Window", self)
        new_window_action.triggered.connect(self.create_new_window)
        file_menu.addAction(new_window_action)

        tile_horizontal_action = QAction("Tile Horizontally", self)
        tile_horizontal_action.triggered.connect(self.tile_horizontally)
        window_menu.addAction(tile_horizontal_action)

        tile_vertical_action = QAction("Tile Vertically", self)
        tile_vertical_action.triggered.connect(self.tile_vertically)
        window_menu.addAction(tile_vertical_action)
    
    def display_message(self, message):
        # Hide any previous message
        self.message_label.hide()
        # Update the QLabel with the new message and show it
        self.message_label.setText(message)
        self.message_label.show()    
    
    def update_mdi_area(self):
        self.mdi_area.update()
        
    def toggle_trendline_mode(self):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow is None:
            return
        self.canvas = active_subwindow.widget()
        if self.canvas and self.canvas.line_drawer:
            self.canvas.stop_line_drawing()
        elif self.canvas:
            self.canvas.start_line_drawing()
    
    def create_new_window(self, file_path, num_bars):
        
        sub_window = QMdiSubWindow(self)
        self.canvas = MplCanvas(file_path, num_bars, self)
        sub_window.setWidget(self.canvas) 
        sub_window.setWindowTitle(file_path.split("/")[-1])
        self.mdi_area.addSubWindow(sub_window)  # Use self.mdi_area instead of self.mdi
        sub_window.show()
    
    def tile_subwindows(self, subwindow):  # Add the subwindow as an argument
        self.mdi_area.removeSubWindow(subwindow)  # Remove the subwindow
        self.mdi_area.tileSubWindows()  # Tile the remaining subwindows
        
    def tile_horizontally(self):
        sub_window_list = self.mdi_area.subWindowList()
        window_count = len(sub_window_list)
        if window_count == 0:
            return

        mdi_area_width = self.mdi_area.width()
        mdi_area_height = self.mdi_area.height()
        window_width = mdi_area_width // window_count

        for index, sub_window in enumerate(sub_window_list):
            sub_window.setGeometry(index * window_width, 0, window_width, mdi_area_height)

    def tile_vertically(self):
        sub_window_list = self.mdi_area.subWindowList()
        window_count = len(sub_window_list)
        if window_count == 0:
            return

        mdi_area_width = self.mdi_area.width()
        mdi_area_height = self.mdi_area.height()
        window_height = mdi_area_height // window_count

        for index, sub_window in enumerate(sub_window_list):
            sub_window.setGeometry(0, index * window_height, mdi_area_width, window_height)
    
    def open_chart(self):
        self.chart_dialog = ChartDialog(self)
        if self.chart_dialog.exec_() == QDialog.Accepted:
            file_path = self.chart_dialog.tab1.file_path
            number_of_bars = self.chart_dialog.tab1.num_bars.value()  # Get the value from the spinbox
            self.create_new_window(file_path, number_of_bars)  # Pass the number of bars to create_new_window
    
    def change_chart_type(self, chart_type):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            active_subwindow.widget().change_chart_type(chart_type)
    
    def change_indicator_type(self, indicator_type):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            active_subwindow.widget().change_indicator_type(indicator_type)
       
    def change_edit_type(self, edit_type):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            active_subwindow.widget().change_edit_type(edit_type)
            
    def refresh_chart(self):
        active_subwindow = self.mdi_area.activeSubWindow()
        if active_subwindow:
            active_subwindow.widget().refresh_chart()
           
          
            

    def instrument_function(self):
        self.display_message("Instrument function triggered")  # This is a placeholder, replace with your actual code
        # Your code here...   

    def strategy_function(self):
        self.display_message("Strategy action function triggered")  # This is a placeholder, replace with your actual code
        # Your code here...       

    def strategy_builder_function(self):
        self.display_message("Strategy builder action function triggered")  # This is a placeholder, replace with your actual code
        # Your code here...       

    def grid_display_function(self):
        self.display_message("Grid display function triggered")  # This is a placeholder, replace with your actual code
        # Your code here...  

    def scanner_function(self):
        self.display_message("Scanner function triggered")  # This is a placeholder, replace with your actual code
        # Your code here... 
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mdi_window = MdiWindow()
    mdi_window.show()
    #create_json()
    sys.exit(app.exec_())