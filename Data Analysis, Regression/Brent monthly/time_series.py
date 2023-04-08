import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

class TimeSeries():
    def __init__(self, path: str, filename: str) -> None:
        '''
        Class for file parsing, plotting, moving average calculation,
        and regression analysis for time series data

        Parameters:
        ----------
        path : str
            The input data file path
        filename: str
            The name of data file to be used as plot title

        Variables:
        -----------
        self.title: str
            Stores the name of dataset (removes the .csv extension)
        self.data: numpy array
            Stores the data from datafile in a  numpy array
        self.row_num: int
            Stores the number of rows in datafile
        self.col_num: int
            Stores the number of columns in datafile

        Returns:
        -------
        None.

        '''
        self.title = filename[:-4]
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1)
        self.row_num, self.col_num = np.shape(self.data)
        self.data[:,0] = np.linspace(0, self.row_num-1, self.row_num)

    def plot_data(self, col = 1) -> None:
        '''
        Plots the time series data.

        Paramters:
        ----------
        col: int
            Specifies the column with which to plot the graph

        Variables:
        ----------
        figure1, ax: matplotlib object
            The subplot objects
        
        Returns
        -------
        None.

        '''
        figure1, ax = plt.subplots()
        self.data[:, 0] = np.arange(self.row_num)
        ax.plot(self.data[:, 0], self.data[:, col])

        ax.set( title= self.title, 
            xlabel="Month", ylabel="Brent Crude Spot Price ($)");

    def moving_average(self, m: int = 3) -> np.ndarray:
        '''
        Responsible for computing the moving_average of the target 
        variable.

        Parameters
        ----------
        m : int
            Window of moving average, set to 3 by default
        figure1, ax: matplotlib object
            The subplot objects

        Variables:
        ----------
        figure2, bx: matplotlib object
            The subplot objects

        Returns
        -------
        moving_average : np.ndarray
            A Numpy array that contains the sequential moving 
            average values.

        '''
        moving_average = np.convolve(self.data[:, 1], np.ones(m), 'valid') / m

        figure2, bx = plt.subplots()

        k = int((m-1)/2)
        bx.plot(self.data[k:-k, 0], self.data[k:-k, 1],
                color = 'blue', label = "Raw_Data")                
        bx.plot(self.data[k:-k, 0], moving_average, color = 'red',
                linestyle = "dashdot", label = "Moving_Average")
        bx.legend()
        bx.set(title=self.title, 
           xlabel="Month", ylabel="Brent Crude Spot Price ($)");       
        return moving_average

    def linear_reg_y0(self, y : np.ndarray, m : int = 3) -> np.ndarray:
        '''
        A function to plot the linear regression model for the given 
        time series dataset. This function has y-int = 0.

        Parameters
        ----------
        y : np.ndarray
            A Numpy array that contains the sequential moving 
            average values.
        m : Window of moving average, set to 3 by default

        Variables:
        ----------
        figure3, cx: matplotlib object
            The subplot objects

        Returns
        -------
        B : Numpy Array
            An array consisting of slope of each value in
            linear equation y = mx
        R : Numpy Array
            An array consisting of Residuals of the model

        '''
        k = int((m-1)/2)
        B, R, x, _ = np.linalg.lstsq(self.data[k:-k,0,np.newaxis],
                                            y, rcond = None)
        figure3, cx = plt.subplots()
        cx.plot(self.data[k:-k,0], y, 'o', label='Raw_Data',
                linestyle = "solid", markersize=10)
        cx.plot(self.data[k:-k,0], B*self.data[k:-k,0],
                'r', linestyle = "dotted",label='Plot: y-Int = 0')
        cx.legend()
        cx.set(title="Model : y Intercept = 0", 
           xlabel="Month", ylabel="Brent Crude Spot Price ($)"); 
        return B, R
    
    def linear_reg_yInt(self, y : np.ndarray, m : int = 3) -> np.ndarray:
        '''
        A function to plot the linear regression model for the given 
        time series dataset. This function has y-int != 0.

        Parameters
        ----------
        y : np.ndarray
            A Numpy array that contains the sequential moving 
            average values.
        m : Window of moving average, set to 3 by default

        Variables:
        ----------
        figure4, dx: matplotlib object
            The subplot objects

        Returns
        -------
        B : Numpy Array
            An array consisting of slope of each value in
            linear equation y = mx + c
        R : Numpy Array
            An array consisting of Residuals of the model

        '''
        k = int((m-1)/2)
        X = np.full((self.row_num-(2*k),2),1)
        X[:,1] = self.data[k:-k,0]
        B, R, x, _ = np.linalg.lstsq(X, y, rcond = None)
        figure4, dx = plt.subplots()
        dx.plot(self.data[k:-k,0], y, 'o', linestyle = "solid",
                label='Raw_Data', markersize=10)
        dx.plot(self.data[k:-k,0], B[0] + B[1]*self.data[k:-k,0],
                'r', linestyle = "dashed", label='Plot: y= mx + c')
        dx.legend()
        dx.set(title="Model: y-intercept = c",
           xlabel="Month", ylabel="Brent Crude Spot Price ($)"); 
        return B, R
    
    def better_model(self) -> str:
        """
        A function to verify which model performs better, we have set m= 11.

        Variables:
        ----------
        B_y0 : Numpy Array
            An array consisting of slopes of model 1
        R_y0 : Numpy Array
            An array consisting of Residuals of the model 1
        B : Numpy Array
            An array consisting of slopes of model 2
        R : Numpy Array
            An array consisting of Residuals of the model 2
        
        Return:
        --------
        None [print statement]
        """
        data_mov_avg = self.moving_average(11)
        B_y0, R_y0 = self.linear_reg_y0(data_mov_avg, 11)
        B, R = self.linear_reg_yInt(data_mov_avg, 11)
        
        if R_y0 < R:
            return "The first model performs better"
        elif R == R_y0 :
            return "There is no difference in performance"
        else:
            return "The second model performs better"

if __name__ == "__main__":   
    input = r"brent_monthly.csv"
    timeseries = TimeSeries(input, str(Path(input).name) )
    timeseries.plot_data()
    timeseries.better_model()

   