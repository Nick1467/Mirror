"""

    CCU QEL hdf5 file reader, for data processing of Labber measured data.

functions
----------
    get path -- Pop up a dialog to browse a file path then return it.
    f2ind --
    i2ind --
    get_fdata_and_idata --

namespcae: VNAxDC
----------
    get_info -- Get informations about measurment, e.g. n_pts, startf, stopf etc... .
    get_data_and_bg -- Get data and background data of VNA traces.
    get_ploting_objs --  Return figure, axes, extent that auto sets based on `info`.
    add_debg_file -- Add an hdf5 file for debackground data that log browser can open.
"""

import h5py
from matplotlib import pyplot as plt
import numpy as np

__all__ = [
    'get_path',
    'VNAxDC',
    'f2ind',
    'i2ind',
    'get_idata',
    'get_fdata'
]

def get_path(ext: str, title = 'Select a file path', save_file = False):
    """Pop up a dialog to browse a file path then return it.

    Argumenumt
    ----------
    ext : string
        The filename extestion.
    title : string, optional
        Default is 'Select a file path', the title displayed on the dialog.
    save_file : bool, optioanl
        Default is False. For it's True, it ask a path to save a file.
    
    Return
    ----------
    filepath : string
        the selected filepath, empty string when canceled.
    """
    from tkinter import Tk, filedialog
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if save_file:
        dialog = filedialog.asksaveasfilename
    else:
        dialog = filedialog.askopenfilename
    filepath = dialog(filetypes = [(ext.upper() + ' Files', ext)],
                      title = title)
    return filepath

def f2ind(info, frqe):
    """return index of the element in VNA sweeped frequencies that is closest to `freq`."""
    f0 = info['VNA - start frequency']
    f1 = info['VNA - stop frequency']
    nf = info['VNA - # of points']
    step = (f1 - f0) / nf
    if (frqe < f0 and frqe < f1) or\
       (frqe > f0 and frqe > f1):
        raise Exception(
            f'specified frequency `{frqe}` is out of range' ) from None

    return round( (frqe - f0) / step)

def i2ind(info, current):
    """return index of the element in DC sweeped currents that is closest to `current`."""
    for s in VNAxDC.dc_supply_no:
        try:
            if info[f'DC{s} - sweep']: 
                i0 = info[f'DC{s} - start current']
                i1 = info[f'DC{s} - stop current']
                ni = info[f'DC{s} - # of steps']
                step = (i1 - i0) / ni
                sweeped_dc_no = s
        except KeyError as e:
            pass
    if sweeped_dc_no == '':
        raise Exception('No sweeped quantity') from None
    if (current < i0 and current < i1) or\
       (current > i0 and current > i1):
        raise Exception(
            f'specified current `{current}` is out of range' ) from None
    
    return round( (current - i0) / step)


def get_idata(info):


    ntrace = info['VNA - # of traces']
    
    sweeped_dc_no = ''
    for s in VNAxDC.dc_supply_no:
        try:
            if info[f'DC{s} - sweep']: 
                starti = info[f'DC{s} - start current']
                stopi = info[f'DC{s} - stop current']
                sweeped_dc_no = s
        except KeyError as e:
                pass
    if sweeped_dc_no == '':
        raise Exception('No sweeped quantity')
    return np.linspace(starti, stopi, ntrace)

def get_fdata(info):
    # VNA
    startf = info['VNA - start frequency']
    stopf = info['VNA - stop frequency']
    nf = info['VNA - # of points']
    return np.linspace(startf, stopf, nf)



class VNAxANY:
    """A namespace for gneral experimental setup with VNA sweep
    
    """
    def __init__(self) -> None:
        raise Exception('Creating object for `AxB` is not allowed.')
    

    @staticmethod
    def get_info(hdf_filepath: str, print_info = False):
        """Get informations about measurment, e.g. n_pts, startf, stopf etc... .
        
        Argumenumt
        ----------
        hdf_filepath : string
            The filename for measurement data.
        print_info : bool, optioanl
            Default is False, print the obtained values.
        
        Return
        ----------
        info : dictionary
            A dictionary contains information about measurment, see code for details.
        """
        info = {}
        with h5py.File(hdf_filepath, 'r') as f:
            # NVA: Find which trace it has measured
            trace = ''
            for trial_trace in VNAxDC.possible_traces:
                try:
                    f['Traces'][f'VNA - {trial_trace}_N']
                    trace = trial_trace
                    info['VNA - trace'] = trace
                except KeyError as e:
                    pass
            if trace == '':
                print("Error: no trace")
                return -1
            # NVA: Get number of traces, points; start, stop frequency
            npts = f['Traces'][f'VNA - {trace}_N'][0]
            ntrc = len(f['Traces'][f'VNA - {trace}'][0, 0, :])
            startf, sepf = f['Traces'][f'VNA - {trace}_t0dt'][0, :]
            stopf = startf + sepf * (npts-1)
            info['VNA - # of points'] = npts
            info['VNA - # of traces'] = ntrc
            info['VNA - start frequency'] = startf
            info['VNA - stop frequency'] = stopf

        if print_info:
            for key, value in info.items():
                print(key, ':', value)
        return info

    @staticmethod
    def get_data(data_filepath: str):
        """Return all trace into as a 2d np array."""
        info = VNAxDC.get_info(data_filepath, print_info = False)
        ntrc = info['VNA - # of traces']
        npts = info['VNA - # of points']
        trace = info['VNA - trace']

        with h5py.File(data_filepath, 'r') as f:
            data = np.zeros([ntrc, npts], complex)
            s21_dataset = f['Traces'][f'VNA - {trace}']
            data = s21_dataset[:, 0, :] + s21_dataset[:, 1, :]*1j

        return data


    @staticmethod
    def add_debg_file(data_filepath: str, bg_filepath: str, mode = '/'):
        """Add an hdf5 file for debackground data that log browser can open.

        mode: str, optional
            Support `-` or `/`, the debackground method. However, we believe `/`
            is the coorect method.
        
        It creats a copy of `data_filepath`, then rewrite the VNA trace data.
        """
        import os
        import shutil
        def append_debackground_to_file_path(file_path: str) -> str:
            base_path, filename = os.path.split(file_path)
            filename_without_extension, file_extension = os.path.splitext(filename)
            new_filename = f"{filename_without_extension}_debackground{file_extension}"
            new_file_path = os.path.join(base_path, new_filename)
            return new_file_path

        data = VNAxDC.get_data(data_filepath)
        bg = VNAxDC.get_data(bg_filepath)
        extended_bg = bg * np.ones_like(data)
        info = VNAxDC.get_info(data_filepath)

        if mode == '-':
            is_snn = info['VNA - trace'][1] == info['VNA - trace'][2]
            if is_snn:
                debg_data = data - extended_bg
            else:
                # If you don't know why there are 1+, you are soooo stupid
                debg_data = 1 + data - extended_bg
        elif mode == '/':
            debg_data = data / extended_bg

        # creat a new file and write debackground data
        trace = info['VNA - trace']
        debg_data_filepath = append_debackground_to_file_path(data_filepath)
        shutil.copyfile(data_filepath, debg_data_filepath)
        with h5py.File(debg_data_filepath, 'r+') as f:
            # labber uses float16 to store data, np uses float128 by default
            f['Traces'][f'VNA - {trace}'][:, 0, :] = np.real(debg_data).astype(np.float16)
            f['Traces'][f'VNA - {trace}'][:, 1, :] = np.imag(debg_data).astype(np.float16)
        return debg_data_filepath

class VNAxDC:
    """A namespace for functions for VNA with a DC sweep experimental setup
    

    """
    def __init__(self) -> None:
        raise Exception('Creating object for `VNAxDC` is not allowed.')

    dc_supply_no = ['1', '2', '3', '4']
    possible_traces = [
            'S11', 'S12', 'S13', 'S14',
            'S21', 'S22', 'S23', 'S24',
            'S31', 'S32', 'S33', 'S34',
            'S41', 'S42', 'S43', 'S44',
    ]

    @staticmethod
    def get_info(hdf_filepath: str, print_info = False):
        """Get informations about measurment, e.g. n_pts, startf, stopf etc... .
        
        Argumenumt
        ----------
        hdf_filepath : string
            The filename for measurement data.
        print_info : bool, optioanl
            Default is False, print the obtained values.
        
        Return
        ----------
        info : dictionary
            A dictionary contains information about measurment, see code for details.
        """

        info = {}
        with h5py.File(hdf_filepath, 'r') as f:
            # NVA: Find which trace it has measured
            trace = ''
            for trial_trace in VNAxDC.possible_traces:
                try:
                    f['Traces'][f'VNA - {trial_trace}_N']
                    trace = trial_trace
                    info['VNA - trace'] = trace
                except KeyError as e:
                    pass
            if trace == '':
                print("Error: no trace")
                return -1
            # NVA: Get number of traces, points; start, stop frequency
            npts = f['Traces'][f'VNA - {trace}_N'][0]
            ntrc = len(f['Traces'][f'VNA - {trace}'][0, 0, :])
            startf, sepf = f['Traces'][f'VNA - {trace}_t0dt'][0, :]
            stopf = startf + sepf * (npts-1)
            info['VNA - # of points'] = npts
            info['VNA - # of traces'] = ntrc
            info['VNA - start frequency'] = startf
            info['VNA - stop frequency'] = stopf

            # DC: 
            for s in VNAxDC.dc_supply_no:
                try:
                    dc_step_item = f['Step config'][f'DC supply - {s} - Current']['Step items']
                    # step_item[0][:] = [range_type, step_type, single, start, stop, center, 
                    #                    span, step, n_pts, interp, sweep_rate]
                    # range type: singel point -> 0, sweep-> 1
                    sweep = dc_step_item[0][0]
                    starti = dc_step_item[0][3]
                    stopi = dc_step_item[0][4]
                    # if stopi < starti: 
                    #     starti, stopi = stopi, starti
                    n_pts = dc_step_item[0][8]
                    if sweep:
                        info[f'DC{s} - sweep'] = True
                        info[f'DC{s} - start current'] = starti
                        info[f'DC{s} - stop current'] = stopi
                        info[f'DC{s} - # of steps'] = n_pts 
                    else:
                        info[f'DC{s} - sweep'] = False
                        info[f'DC{s} - current'] = starti
                except KeyError as e:
                    pass

            if print_info:
                for key, value in info.items():
                    print(key, ':', value)
        return info


    @staticmethod
    def get_data(data_filepath: str):
        """ Get data of VNA traces.

        Arguments
        ----------
        data_filepath: string
            file path of measurement data.
        flip: bool, optional
            Default is True, flip the axis to display it coorectly.

        Return
        ----------
        data: ndarray
            measurment data, by VNA.
        """
        info = VNAxDC.get_info(data_filepath, print_info = False)
        ntrc = info['VNA - # of traces']
        npts = info['VNA - # of points']
        trace = info['VNA - trace']

        with h5py.File(data_filepath, 'r') as f:
            data = np.zeros([ntrc, npts], complex)
            s21_dataset = f['Traces'][f'VNA - {trace}']
            data = s21_dataset[:, 0, :] + s21_dataset[:, 1, :]*1j

        return data

    @staticmethod
    def get_ploting_objs(info: dict, transpose = True):
        """ Return figure, axes, extent that auto sets based on `info`. use plt.imshow() to plot.

        Arguments
        ----------
        info: dictionary
            A dictionary contains information about measurment.
        transpose: optional, default is True
            If true, the VNA frequency will be on y axis instead of x axis.
        Returns
        ----------
        fig: matplotlib.figure.Figure
            The figure object.
        ax: matplotlib.axes.Axes
            The axes object.
        extend: list
            used as and argument of plt.imshow().
        """
        # Gather nessesay informations
        startf = info['VNA - start frequency']
        stopf = info['VNA - stop frequency']
        trace = info['VNA - trace']
        sweeped_dc_no = ''
        for s in VNAxDC.dc_supply_no:
            try:
                if info[f'DC{s} - sweep']: 
                    starti = info[f'DC{s} - start current']
                    stopi = info[f'DC{s} - stop current']
                    sweeped_dc_no = s
            except KeyError as e:
                    pass
        if sweeped_dc_no == '':
            raise Exception('No sweeped quantity')

        # Check whether the sweep quantity is flipped
        iflip, fflip = False, False
        if starti > stopi:
            iflip = True
        if startf > stopf:
            fflip = True
    
        # cearting fig and ax and apply settings
        fig, ax = plt.subplots()
        if not fflip: f0, f1 = startf, stopf
        else: f0, f1 = stopf, startf
        if not iflip: i0, i1 = starti, stopi
        else: i0, i1 = stopi, starti
        if transpose:
            extent = [i0*1e3, i1*1e3, f0/1e9, f1/1e9]
            ax.set_title(trace)
            ax.set_xlabel(f'DC{s} current / mA')
            ax.set_ylabel('VNA frequency / GHz')
        else:
            extent = [f0/1e9, f1/1e9, i0*1e3, i1*1e3]
            ax.set_title(trace)
            ax.set_ylabel(f'DC{s} current / mA')
            ax.set_xlabel('VNA frequency / GHz')

        def flipfunc(data):
            """flip the data to plot correctly
            
            We often make a plot that x and y axis that acceding from left to right
            and from bottom to up. However, the data is stored in an array from
            left to right and from up to down, so we need to flip axis 0.

            Futuremore, if our measurment swpeed the quantity from large to small,
            the stored data will be flipped again, so we need to flip the correpond
            axis, in order to make a plot we want.

            If we make the plot transpose, then not only we transpose the data, we need 
            to oppsite the flip rule for x and y axis.
            """
            if transpose:
                if not fflip:
                    data = np.flip(data, axis=0)
                if iflip:
                    data = np.flip(data, axis=1)
            if not transpose:
                data = data.T
                if fflip:
                    data = np.flip(data, axis=1)
                if not iflip:
                    data = np.flip(data, axis=0)
            return data

        return fig, ax, extent, flipfunc

    @staticmethod
    def add_debg_file(data_filepath: str, bg_filepath: str,  filename:str = None, mode:str = '/'):
        """Add an hdf5 file for debackground data that log browser can open.

        
        filename: str, optional
            Default is None, then the filename will be original file name with _debackground suffix.
        mode: str, optional
            Support `-` or `/`, the debackground method. However, we believe `/`
            is the coorect method.
        
            
        It creats a copy of `data_filepath`, then rewrite the VNA trace data.
        """
        import os
        import shutil
        def append_debackground_to_file_path(file_path: str) -> str:
            base_path, filename = os.path.split(file_path)
            filename_without_extension, file_extension = os.path.splitext(filename)
            new_filename = f"{filename_without_extension}_debackground{file_extension}"
            new_file_path = os.path.join(base_path, new_filename)
            return new_file_path

        data = VNAxDC.get_data(data_filepath)
        bg = VNAxDC.get_data(bg_filepath)
        extended_bg = bg * np.ones_like(data)
        info = VNAxDC.get_info(data_filepath)

        if mode == '-':
            is_snn = info['VNA - trace'][1] == info['VNA - trace'][2]
            if is_snn:
                debg_data = data - extended_bg
            else:
                # If you don't know why there are 1+, you are soooo stupid
                debg_data = 1 + data - extended_bg
        elif mode == '/':
            debg_data = data / extended_bg

        # creat a new file and write debackground data
        trace = info['VNA - trace']
        if filename:
            base_path, filename = os.path.split(data_filepath)
            debg_data_filepath = os.path.join(base_path, filename)
        else:
            debg_data_filepath = append_debackground_to_file_path(data_filepath)
        shutil.copyfile(data_filepath, debg_data_filepath)
        with h5py.File(debg_data_filepath, 'r+') as f:
            # labber uses float16 to store data, np uses float128 by default
            f['Traces'][f'VNA - {trace}'][:, 0, :] = np.real(debg_data).astype(np.float16)
            f['Traces'][f'VNA - {trace}'][:, 1, :] = np.imag(debg_data).astype(np.float16)
        return debg_data_filepath