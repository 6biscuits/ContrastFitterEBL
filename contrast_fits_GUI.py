#!/usr/bin/env python
# coding: utf-8

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Script for the import of .csv or .xlsx data to produce contrast 
curves and fitted contrast values.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  '''


import numpy as np
import pandas as pd
# import xlsxwriter # type: ignore
import math
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning) # type: ignore
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Global dictionary to store column lists
columns_as_lists = {}
header_names = []
num_calls = 0
FILE_PATH = None

# global flag for active plot
PLOT_ACTIVE = False

# Global matplotlib settings
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.size"] = 10

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

The contrast curve describes the remaining resist fraction of a 
uniformly illuminated resist versus the logarithm of the applied 
exposure dose.

contrast = gamma = 1 / [ log10(D100 / D0) ]
    D100 is the dose for FULL resist removal (linearised).
    D0   is the dose for NO   resist removal (linearised).

Example values:
    D0   = 50  mJ/cm2
    D100 = 150 mJ/cm2

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

IBM FITTING METHOD [fit_function='Ocola (IBM)']
We use the empirical technique by Leo Ocola (IBM, 2023).

    NRT = C0 - exp[S * (D - Dc)]

then, 
    contrast = ln(10) * S * Dc

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

CMTF FITTING METHOD [fit_function='linear']
From Devin Brown's 2023 Georgia Tech presentation.

    CMTF = (D100 - D0) / (D100 + D0) 
         = (10^(1/gamma) - 1) / (10^(1/gamma) + 1)

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

# Define IBM model
def model_function_IBM(D, c0, S, Dc):
    return c0 - np.exp(S * (D - Dc))

# function to fit contrast
def fit_contrast(dose, thickness, fig, resist_name, colour, nrt_cutoff_low=0.04, nrt_cutoff_high=0.5, plot_fit_data_only=False, plot_fit=True, fit_function='both', save_data_to_excel=False, excel_writer=None, print_results_to_terminal=False):

    assert max(thickness) == 1

    # full sorted arrays
    x_sorted = np.linspace(min(dose), max(dose), 1000)

    # Filter out entries where nrt is less than 0.04
    dose_filtered = [x_val for x_val, y_val in zip(dose, thickness) if (y_val > nrt_cutoff_low)]
    thickness_filtered = [y_val for y_val in thickness if (y_val > nrt_cutoff_low)]

    # Sort lists based on dose
    dose_sorted = sorted(dose_filtered)
    thickness_sorted = [x for _,x in sorted(zip(dose_filtered, thickness_filtered))]

    # arrays for fitting
    x_data = np.array(dose_sorted)
    y_data = np.array(thickness_sorted)

    # Plot data
    if plot_fit_data_only:
        fig.scatter(x_data, y_data, label=resist_name, color=colour, s=12, alpha=0.5)
    else:
        fig.scatter(dose, thickness, label=resist_name, color=colour, s=12, alpha=0.5)
    
    # trim arrays to only include the linear part
    x_data_lin = np.array([x_val for x_val, y_val in zip(dose_sorted, thickness_sorted) if (y_val < nrt_cutoff_high)])
    y_data_lin = np.array([y_val for y_val in thickness_sorted if (y_val < nrt_cutoff_high)])

    # perform linear fit (first degree polyfit)
    coefficients = np.polyfit(x_data_lin, y_data_lin, 1)
    m_coef, c_coef = coefficients
    y_fit = m_coef * x_sorted + c_coef

    # calculate D0, D100, CMTF and gamma
    D0 = (1 - c_coef) / m_coef
    D100 = (0 - c_coef) / m_coef
    cmtf = (D100 - D0)/(D100 + D0)
    gamma = 1 / (math.log10(D100 / D0))

    # DO  LINEAR FIT
    if fit_function=='linear' or fit_function=='both':

        # Print the fitted parameters
        if print_results_to_terminal:
            print(f"CMTF fit parameters for {resist_name}\n\tD0       = {D0:.2f}\n\tD100     = {D100:.2f}\n\tCMTF     = {cmtf:.2f}\n\tcontrast = {gamma:.2f}\n")

        # plot fit and dose values
        if plot_fit:
            fig.plot(x_sorted, y_fit, color=colour, linestyle=':', label=fr'CMTF fit: $\gamma={gamma:.2f}$')
            fig.scatter([D100, D0], [0, 1], color=colour, marker='*', s=35)
            fig.text(0.01, 0.1+0.03*num_calls, f'$D_{{100}}=${D100:.0f}\t$D_0=${D0:.0f}', ha='left', color=colour, transform=fig.transAxes)

    # DO IBM FIT
    initial_guess = [1, 0.01, 500] # Initial guess for the parameters
    gamma = 0
    if fit_function=='Ocola (IBM)' or fit_function=='both':
        try:
            # Fit the model to the data
            popt, pcov = curve_fit(model_function_IBM, x_data, y_data, p0=initial_guess)
            c0, S, Dc = popt
            gamma = math.log(10) * S * Dc

            # Print the fitted parameters
            if print_results_to_terminal:
                print(f"IBM fit parameters for {resist_name}\n\tD0       = {D0:.2f}\n\tD100     = {D100:.2f}\n\tc0       = {c0:.2f}\n\tS        = {S:.2f}\n\tDc       = {Dc:.2f}\n\tcontrast = {gamma:.2f}\n")
            
            # Generate fitted y data
            y_fit = model_function_IBM(x_sorted, *popt)
            y_fit_discrete = model_function_IBM(x_data, *popt)
            
            # Plot the fitted curve
            if plot_fit:
                fig.plot(x_sorted, y_fit, color=colour, label=fr'Leo Ocola (IBM) fit:  $\gamma={gamma:.2f}$')
                if fit_function=='Ocola (IBM)':
                    fig.text(0.01, 0.1+0.03*num_calls, f'$D_{{100}}=${D100:.0f}\t$D_0=${D0:.0f}', ha='left', color=colour, transform=fig.transAxes)
            
        except Exception as e:
            print(f"Error in fitting: {e}")

    if fit_function not in ['Ocola (IBM)', 'both', 'linear']:
        raise ValueError('The choices for fit_function input to this function are "Ocola (IBM)", "LINEAR", or "BOTH". Please enter one of these choices when calling.') 
    
    # output fitted IBM values to excel spreadsheet
    if save_data_to_excel:
        # check that fit was performed, if not, do fit
        if not (fit_function=='Ocola (IBM)' or fit_function=='both'):
            popt, pcov = curve_fit(model_function_IBM, x_data, y_data, p0=initial_guess)
            c0, S, Dc = popt
            gamma = math.log(10) * S * Dc
            y_fit_discrete = model_function_IBM(x_data, *popt)

        # Write to Excel
        df_out = pd.DataFrame()
        df_out['Dose [µC/cm2]'] = x_data        # Copy sorted dose values
        df_out['Measured NRT'] = y_data         # Copy measured NRT values
        df_out['Fitted NRT'] = y_fit_discrete   # Copy fitted values

        # Convert sheet name if necessary
        # sheet_name = f'{resist_name}, contrast={gamma:.1f}'
        sheet_name = f'{resist_name}'
        chars_to_replace = "[]:*?/\\"
        replacement_char = "|"
        translation_table = str.maketrans(chars_to_replace, replacement_char * len(chars_to_replace))
        sheet_name_safe = sheet_name.translate(translation_table)

        # Write
        df_out.to_excel(excel_writer, sheet_name=sheet_name_safe, index=False)

        # print(f'Data saved to Excel: {excel_writer}')


''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
GUI FUNCTIONS AND LAYOUT

We use Tkinter for the GUI.

Here we define functions for various buttons in the layout, then 
place Tkinter objects onto a layout grid.
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''


# handle upload
def handle_upload(file_path, fit_type, print_lists=False, sort_lists=True, plot_data=True, contrast_fit=True, save_plot=False, save_excel=False):

    global columns_as_lists
    global header_names
    global num_calls
    global excel_path
    
    # # update options based on GUI
    # save_excel = print_lists_opt.get()
    # save_plot = save_plot_opt.get()

    FILE_PATH = file_path

    # Clear previous data
    columns_as_lists.clear()
    header_names.clear()
    num_calls = 0

    # Get the file name and path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = os.path.basename(FILE_PATH)
    file_name_no_ext = file_name.rsplit('.', 1)[0]


    print('File path: ' + os.path.realpath(__file__))

    # setup plot filename
    plot_filename = f'{file_name_no_ext}_fitted.png'
    plot_path = os.path.join(dir_path, 'fitted_plots', plot_filename)

    # setup writer for excel output
    excel_filename = f'{file_name_no_ext}_fitted.xlsx'
    excel_path = os.path.join(dir_path, 'fitted_values', excel_filename)
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    # print selected value and file
    print(f'\n\nDoing fit\nSelected fitting type: {fit_type}')
    print(f'Selected file:         {file_name}\n\n')
    
    # Read the file into a DataFrame
    if file_name.endswith('.csv'):
        df = pd.read_csv(FILE_PATH)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df = pd.read_excel(FILE_PATH)
    else:
        raise ValueError('Unsupported file type: only .csv, .xlsx or .xls files supported currently.')
        return
    
    # Get the header names
    header_names = df.columns.tolist()
    
    # Save columns into lists
    columns_as_lists = {header: df[header].tolist() for header in header_names}

    # Convert data into floats
    columns_as_lists_of_floats = {k:list(map(float, columns_as_lists[k])) for k in columns_as_lists}
    columns_as_lists = columns_as_lists_of_floats

    # Display the headers and the first few elements of each column list
    if print_lists:
        if not columns_as_lists:
            print("No data to print. Please upload a file first.")
            return
        
        for header, column_list in columns_as_lists.items():
            print(f"{header}\n{column_list[:5]}\n")
    
    # sort lists
    if sort_lists:
        if not columns_as_lists:
            print("No data to sort. Please upload a file first.")
            return

        # assign dose list
        dose = columns_as_lists[header_names[0]]

        # assign film thickness lists
        fts = []
        num_resists = len(columns_as_lists)-1
        for i in range(num_resists):
            fts.append(columns_as_lists[header_names[i+1]])

    # initialise figure
    figure = plt.figure(figsize=(10, 6))
    fig = plt.axes()
    # color_map = plt.cm.get_cmap('gist_earth', num_resists+1) # type: ignore
    color_map = plt.cm.get_cmap(colourmap_choice.get(), num_resists+1) # type: ignore

    '''
    cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral',
            'gist_ncar'])]
    '''

    # figure settings
    plt.ylim([-0.08,1.08])
    plt.xlim([50,1000])
    plt.xscale('log')
    plt.xlabel('Dose [µC/cm2]')
    plt.ylabel('Normalised resist thickness [NRT]')
    plt.title(f'Contrast curve (from: {file_name})')

     # plot and fit data
    if plot_data and contrast_fit:
        if not columns_as_lists:
            print("No data to plot. Please upload a file first.")
            return
        
        for i, ft_data in enumerate(fts):
            fit_contrast(dose, ft_data, fig, header_names[i+1], color_map(i), fit_function=f'{fit_type}', save_data_to_excel=save_excel, excel_writer=writer)
            num_calls += 1
        # TODO: make legend off to the right of the plot, or make a table
        plt.legend()

        # create canvas to place figure
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid()

    # plot only
    elif plot_data and not contrast_fit:
        if not columns_as_lists:
            print("No data to plot. Please upload a file first.")
            return
        
        for i, ft_data in enumerate(fts):
            # plt.scatter(dose, ft_data, label=header_names[i+1], color=color_map(i), s=12, alpha=0.5)
            plt.scatter(dose, ft_data, color=color_map(i), s=12, alpha=0.5) # remove label on raw data (less clutter in the legend)
        plt.legend()

        # create canvas to place figure
        canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid()

    # Display the plot in the Tkinter GUI
    display_plot(figure)

    # close Excel writer
    writer.close()

# to display plot
def display_plot(figure):
    PLOT_ACTIVE = True
    global canvas

    # Clear previous plot if exists
    for widget in plot_frame.winfo_children():
        widget.destroy()
    
    # Create a canvas and add the figure
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().grid()

    # Show the clear button after showing the plot
    clear_plot_button['state'] = 'normal'
    save_plot_button['state'] = 'normal'
    save_excel_button['state'] = 'normal'
    refresh_plot_button['state'] = 'normal'

# Clear plot
def clear_plot():
    global canvas
    if canvas:
        canvas.get_tk_widget().destroy()
        canvas = None
    # Hide the clear button and show the show plot button
    clear_plot_button['state'] = 'disabled'
    save_plot_button['state'] = 'disabled'
    save_excel_button['state'] = 'disabled'
    refresh_plot_button['state'] = 'disabled'
    save_path_label.grid_forget()
    upload_message.config(text=f"Please upload new data...")
    PLOT_ACTIVE = False

# handle closing
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()
        exit()

# upload_file --> handle_upload --> fit_contrast
def upload_file(return_only=False):
    global FILE_PATH
    if not return_only:
        FILE_PATH = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("Old Excel files", "*.xls")])
        FILE_NAME = os.path.basename(FILE_PATH)
        if FILE_PATH:
            handle_upload(FILE_PATH, fit_type.get())
            upload_message.config(text=f"{FILE_NAME} loaded")

# refresh plot
def refresh_plot():
    global FILE_PATH
    handle_upload(FILE_PATH, fit_type.get())
    upload_message.config(text=f"{FILE_NAME} refreshed with new plot settings.")

# Save plot
def save_plot():
    save_path = filedialog.asksaveasfilename(defaultextension='.png')
    print(f'Plot saved at: {save_path}')
    save_path_label.config(text=f"Plot saved at: {save_path}")
    plt.savefig(save_path)

# Save plot
def save_excel():
    global FILE_PATH
    global excel_path
    handle_upload(FILE_PATH, fit_type.get(), save_excel=True)
    print(f'Saved excel sheet at: {excel_path}')
    save_path_label.config(text=f"Excel saved at: {excel_path}")

    

''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- - - - - - - - - - - - - - - GUI LAYOUT- - - - - - - - - - - - -
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- upload button and options section at top
- plotting underneath
- 4 columns in layout
'''

FILE_PATH = ""
FILE_NAME = ""
excel_path = ""

# Create the main application window
root = tk.Tk()
root.title("Contrast Fitting Application")
root.geometry("1000x800")

# Create upload and plot frames
upload_frame = tk.Frame(root)
upload_frame.pack(side=tk.TOP, pady=10)

plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# FIT TYPE (DROPDOWN)
fit_type_label = tk.Label(upload_frame, text="Select fit type:", justify='right', anchor='w')
fit_type_label.grid(row=0, column=0, padx=10, pady=10)
fit_type = tk.StringVar()
fit_type.set("both")  # default value
dropdown = tk.OptionMenu(upload_frame, fit_type, "linear", "Ocola (IBM)", "both")
dropdown.grid(row=0, column=1, padx=10, pady=10)

# COLOUR MAP (DROPDOWN)
fit_type_label = tk.Label(upload_frame, text="Plot colours:", justify='right', anchor='w')
fit_type_label.grid(row=1, column=0, padx=10, pady=10)
colourmap_choice = tk.StringVar()
colourmap_choice.set("tab20b")  # default value
dropdown = tk.OptionMenu(upload_frame, colourmap_choice, "viridis", 'plasma', 'autumn', 'summer', 'winter', 'spring', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Dark2', 'jet', 'rainbow', 'gist_earth', 'nipy_spectral', 'gist_ncar')
dropdown.grid(row=1, column=1, padx=10, pady=10)

# UPLOAD
upload_label = tk.Label(upload_frame, text="Upload a .csv or .xlsx file:", justify='right')
upload_label.grid(row=0, column=2, padx=10, pady=10)
upload_button = ttk.Button(upload_frame, text="Upload", command=upload_file)
upload_button.grid(row=0, column=3, padx=10, pady=10)
upload_message = tk.Label(upload_frame, text="Waiting for upload...")
upload_message.grid(row=1, column=2, padx=10, pady=10)

#TODO: refresh button
refresh_plot_button = ttk.Button(upload_frame, text="refresh plot", command=refresh_plot, state='disabled')
refresh_plot_button.grid(row=1, column=3, padx=10, pady=10)

# CLEAR PLOT
clear_plot_button = ttk.Button(upload_frame, text="Clear Plot", command=clear_plot, state='disabled')
clear_plot_button.grid(row=2, column=3)

# SAVE PLOT
save_plot_button = ttk.Button(upload_frame, text="Save plot", command=save_plot, state='disabled')
save_plot_button.grid(row=2, column=0, padx=10, pady=10)

# SAVE EXCEL
save_excel_button = ttk.Button(upload_frame, text="Save excel", command=save_excel, state='disabled')
save_excel_button.grid(row=2, column=1, padx=10, pady=10)

save_path_label = ttk.Label(upload_frame, text="")
save_path_label.grid(row=3, column=0, columnspan=4)

# run
root.mainloop()

