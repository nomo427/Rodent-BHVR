import classes
import sys
import os
import pandas as pd
import json
if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg
import shutil
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import Subplot
import numpy as np
from bisect import bisect



def main():

    # Set Theme and Print
    sg.theme('BlueMono')
    sg.Print = print

    # Standardize the size of buttons
    button_width = 8

    # Default num_bin values for each interactive heatmap
    default_num_bin_grayscale = 20
    default_num_bin_timescale = 10
    default_num_bin_contour = 20

    # ===================================================== MENU ===================================================== #

    # Define the options for the tabs at top of screen
    menuDef = [
        ['&File', ['&Open', 'Exit',]],
    ]

    # ============================================== IMAGES / DATA TAB =============================================== #

    # Create options column
    tab_plot_col_options = [
        [sg.Text("Type of plot")],

        #[sg.Drop(values=('Movement Plot', 'Heatmap', 'Center Anxiety', 'Wall Distance', 'Speed', 'Custom Regions'),

        [sg.Drop(values=('Movement Plot', 'Heatmap', 'Center Anxiety', 'Wall Distance', 'Speed'),
                 enable_events=True,
                 key='-PLOT-')
        ],

        [sg.Text("Save as")],

        [sg.Drop(values=('.svg', '.jpg', '.png'),
                 enable_events=True,
                 key='-SAVE-IMG-')
        ]
    ]

    # Create plot column
    tab_plot_col_plot = [
        [sg.Image(key='-IMAGE-PLOT-',
                  filename="Images/Blank.png",
                  size=(400,400))
        ]
    ]

    # Place the tab_plot_col_options column by the tab_plot_col_plot column with a vertical separator in between
    tab_plot_layout = [
        [
            sg.Column(tab_plot_col_options),
            sg.VerticalSeparator(),
            sg.Column(tab_plot_col_plot),
        ]
    ]

    # Raw Data tab
    processed_data_headers = {
        '#Snapshot Timestamp': [],
        'Trigger Region Identifier': [],
        'Position.X': [],
        'Position.Y': [],
        'Position.Z': [],
        'Forward.X': [],
        'Forward.Y': [],
        'Forward.Z': [],
        'file_name': [],
        'file_path': [],
        'maze_type': [],
        '.vrsettings': [],
        '.mazesettings': [],
        'drug_applied': [],
        'time_diff': [],
        'total_time': [],
        'total_distance': [],
        'speed': [],
        'avg_speed': [],
        'y_region': [],
        'y_top_time': [],
        'y_center_time': [],
        'y_left_time': [],
        'y_right_time': [],
        'dist_wall_units': [],
        'center_anxiety': [],
        'time_center_anxiety': [],
        'percent_center': [],
    }

    processed_data_values = []

    tab_data_layout = [
        [
            sg.Button("Save As",
                      enable_events=True,
                      key='-SAVE-PROCESSED-DATA-',
                      size=(button_width,1)),
            sg.Text(key="-ENHANCED-BEHAVIORAL-DISPLAY-"),
        ],

        [
            sg.Table(values=processed_data_values,
                     headings=list(processed_data_headers),
                     auto_size_columns=True,
                     vertical_scroll_only=False,
                     display_row_numbers=True,
                     justification="left",
                     key="-PROCESSED-TABLE-",
                     background_color="#172533",
                     text_color="white",
                     enable_events=True,
                     size=(100, 30))
        ]
    ]

    col_grayscale_heatmap = [
        [
            sg.Slider((5, 50),
                      orientation='h',
                      s=(15, 15),
                      default_value=default_num_bin_grayscale,
                      resolution=1,
                      enable_events=True,
                      key="-SLIDER-GRAYSCALE-",
                      disable_number_display=False),
            sg.Text("Title:"),
            sg.Checkbox("",
                        default=False,
                        enable_events=True,
                        key="-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"),
            sg.Text("Save As:"),
            sg.Drop(values=('.svg', '.jpg', '.png'),
                    enable_events=True,
                    key='-SAVE-GRAYSCALE-HEATMAP-')
        ],

        [sg.Canvas(size=(288, 216),
                   key="-INTERACTIVE-HEATMAP-",
                   background_color='white')
        ],
    ]

    col_timescale_heatmap = [
        [
            sg.Slider((2, 25),
                      orientation='h',
                      s=(15, 15),
                      default_value=default_num_bin_timescale,
                      resolution=1,
                      enable_events=True,
                      key="-SLIDER-TIMESCALE-",
                      disable_number_display=False),
            sg.Text("Title:"),
            sg.Checkbox("",
                        default=False,
                        enable_events=True,
                        key="-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"),
            sg.Text("Save As:"),
            sg.Drop(values=('.svg', '.jpg', '.png'),
                    enable_events=True,
                    key='-SAVE-TIMESCALE-HEATMAP-')
        ],

        [sg.Canvas(size=(288, 216),
                   key="-INTERACTIVE-TIMESCALE-HEATMAP-",
                   background_color='white')
        ],
    ]

    col_contour_heatmap = [
        [
            sg.Slider((5, 100),
                      orientation='h',
                      s=(15, 15),
                      default_value=default_num_bin_contour,
                      resolution=1,
                      enable_events=True,
                      key="-SLIDER-CONTOUR-",
                      disable_number_display=False),
            sg.Text("Title:"),
            sg.Checkbox("",
                        default=False,
                        enable_events=True,
                        key="-CHECKBOX-TITLE-CONTOUR-HEATMAP-"),
            sg.Text("Save As:"),
            sg.Drop(values=('.svg', '.jpg', '.png'),
                    enable_events=True,
                    key='-SAVE-CONTOUR-HEATMAP-')
        ],

        [sg.Canvas(size=(288, 216),
                   key="-INTERACTIVE-CONTOUR-HEATMAP-",
                   background_color='white')
        ],
    ]

    tab_heatmap = [
        [
            sg.Column(col_grayscale_heatmap),
            sg.VerticalSeparator(),
            sg.Column(col_timescale_heatmap),
            sg.VerticalSeparator(),
            sg.Column(col_contour_heatmap)
        ]
    ]

    # Group the plot tab with the data tab
    plot_data_tabs = [
        [
            sg.TabGroup([[
                    sg.Tab('Explore', tab_plot_layout),
                    sg.Tab("Heatmaps", tab_heatmap),
                    sg.Tab('Processed Data', tab_data_layout)
            ]])
        ]
    ]

    # ================================================== TREE COLUMN ================================================= #

    # Set treedata variable to sg.TreeData()
    treedata = sg.TreeData()

    # Define the tree column, and add Generate button at the button. Place this in the col_tree layout
    col_tree = [
        [sg.Tree(data=treedata,
                 headings=[],
                 auto_size_columns=True,
                 num_rows=70,
                 row_height=15,
                 col0_width=30,
                 key='-TREE-',
                 show_expanded=True,
                 vertical_scroll_only=False),
        ]
    ]

    # ================================================= TABLE COLUMN ================================================= #

    # Create a column of buttons that can manipulate the metadata table
    row_buttons = [
            sg.Button("Process",
                      size=(button_width,1)),
            sg.Button("Clear All",
                      size=(button_width,1)),
            sg.Button("Remove",
                      size=(button_width,1)),
            sg.Button("Save As",
                      enable_events=True,
                      key='-SAVE-METADATA-',
                      size=(button_width,1)),
            sg.Button("Copy",
                      enable_events=True,
                      key='-COPY-METADATA-',
                      size=(button_width,1)),
    ]

    # Define the headers for the metadata table. Create dictionary with keys and set values equal to empty lists
    headers = {'file_name': [],
               'file_path': [],
               'maze_type': [],
               '.vrsetting': [],
               '.maze': [],
               'drug_applied': [],
               'total_time': [],
               'distance_traveled': [],
               'avg_speed': [],
               'time_center_anxiety': [],
               'percent_time_center_anxiety': [],
               'avg_dist_wall': [],
               'y_top': [],
               'y_center': [],
               'y_left': [],
               'y_right': [],
    }

    # To add custom regions to dictionary of headers, open the json file
    with open("tree_information.json", "r") as infile:

        # Load the data and add each region to headers dictionary
        data = json.load(infile)
        if len(data["appropriate_custom_regions"]) > 0:
            for region in data["appropriate_custom_regions"]:
                headers[region] = []

    # Make empty df with headers dictionary. This will contain raw data from experiment and be displayed in data tab
    summary_stats_master_df = pd.DataFrame(headers)

    # In order to create the GUI table, grab the Headings and the Values
    tableHeadings = list(headers)
    tableValues = summary_stats_master_df.values.tolist()

    # Define the metadata table row
    row_metadata_table = [sg.Table(values=tableValues,
                                   headings=tableHeadings,
                                   auto_size_columns=True,
                                   vertical_scroll_only=False,
                                   display_row_numbers=True,
                                   justification="left",
                                   key="-TABLE-",
                                   background_color="#172533",
                                   text_color="white",
                                   enable_events=True,
                                   def_col_width=13,
                                   size=(200,40))
    ]

    # Group row_buttons, row_metadata_table, and row_custom_regions in the col_table layout
    col_table = [
        row_buttons,
        row_metadata_table,
    ]

    # =============================================== FINALIZE LAYOUT ================================================ #

    # Put all the columns, rows, and separators together in the layout
    right_col = [[sg.Column(plot_data_tabs, key='-TAB-IMAGE-')],
                 [sg.HorizontalSeparator()],
                 [sg.Column(col_table)]
    ]

    # Finalize layout variable
    layout = [
        [sg.Menu(menuDef, tearoff=True)],
        [sg.Column(col_tree), sg.VerticalSeparator(), sg.Column(right_col)],
    ]

    # Create the window with the layout
    window = sg.Window("Rodent-BHVR v1.0.0",
                       layout,
                       resizable=True,
                       finalize=True)

    # Maximize the window
    window.maximize()

    # ==================================== IMPORTANT GLOBAL VARIABLES & FUNCTIONS ==================================== #


    # ======== IMPORTANT VARIABLES AVAILABLE EVERYWHERE BELOW


    # Canvas figures for each interactive heatmap
    fig_grayscale = None
    fig_timescale = None
    fig_contour = None

    # Convert extensions to Matplotlib arguments
    img_types = {
        '.svg': 'SVG',
        '.jpg': 'JPG',
        '.png': 'PNG',
    }

    # List of all custom regions present within the experiment options to choose from
    custom_regions_list = []

    # Contains the current selected experiment in the table and title of the current selected experiment
    index_current_experiment = None

    # By default, set plot_type to "movement_plot" to automatically display movement plots when an experiment is clicked
    plot_type = "_movement_plot"

    # Create an empty dictionary to hold all the raw data dataframes for each experiment that is generated.
    all_exp_dfs_dict = {}

    # Path to Images Folder
    path_to_images_folder = os.path.dirname(os.path.abspath("Images/Blank.png"))

    # Create Dictionary of all mazes and their boundaries by loading the json file
    try:
        with open('maze_boundaries.json', 'r') as fp:
            all_maze_boundaries = json.load(fp)

    except FileNotFoundError:
        print("something wrong with filename")

    # These strings help to create folder and file icons in the tree column
    folder_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABnUlEQVQ4y8WSv2rUQRSFv7vZgJFFsQg2EkWb4AvEJ8hqKVilSmFn3iNvIAp21oIW9haihBRKiqwElMVsIJjNrprsOr/5dyzml3UhEQIWHhjmcpn7zblw4B9lJ8Xag9mlmQb3AJzX3tOX8Tngzg349q7t5xcfzpKGhOFHnjx+9qLTzW8wsmFTL2Gzk7Y2O/k9kCbtwUZbV+Zvo8Md3PALrjoiqsKSR9ljpAJpwOsNtlfXfRvoNU8Arr/NsVo0ry5z4dZN5hoGqEzYDChBOoKwS/vSq0XW3y5NAI/uN1cvLqzQur4MCpBGEEd1PQDfQ74HYR+LfeQOAOYAmgAmbly+dgfid5CHPIKqC74L8RDyGPIYy7+QQjFWa7ICsQ8SpB/IfcJSDVMAJUwJkYDMNOEPIBxA/gnuMyYPijXAI3lMse7FGnIKsIuqrxgRSeXOoYZUCI8pIKW/OHA7kD2YYcpAKgM5ABXk4qSsdJaDOMCsgTIYAlL5TQFTyUIZDmev0N/bnwqnylEBQS45UKnHx/lUlFvA3fo+jwR8ALb47/oNma38cuqiJ9AAAAAASUVORK5CYII='
    file_icon = b'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsSAAALEgHS3X78AAABU0lEQVQ4y52TzStEURiHn/ecc6XG54JSdlMkNhYWsiILS0lsJaUsLW2Mv8CfIDtr2VtbY4GUEvmIZnKbZsY977Uwt2HcyW1+dTZvt6fn9557BGB+aaNQKBR2ifkbgWR+cX13ubO1svz++niVTA1ArDHDg91UahHFsMxbKWycYsjze4muTsP64vT43v7hSf/A0FgdjQPQWAmco68nB+T+SFSqNUQgcIbN1bn8Z3RwvL22MAvcu8TACFgrpMVZ4aUYcn77BMDkxGgemAGOHIBXxRjBWZMKoCPA2h6qEUSRR2MF6GxUUMUaIUgBCNTnAcm3H2G5YQfgvccYIXAtDH7FoKq/AaqKlbrBj2trFVXfBPAea4SOIIsBeN9kkCwxsNkAqRWy7+B7Z00G3xVc2wZeMSI4S7sVYkSk5Z/4PyBWROqvox3A28PN2cjUwinQC9QyckKALxj4kv2auK0xAAAAAElFTkSuQmCC'

    # Name of json file to hold tree file
    tree_filename = 'tree_information.json'


    # ======== IMPORTANT FUNCTIONS


    def add_files_in_folder(parent, dirname):
        """
        add_files_in_folder():  Loads all behavioral files into the tree file structure
        Input:
            parent (string):    Name of parent directory, everything that comes before filename to make an absolute path
            dirname (string):   Name of directory to load
        Output:
            None
        """

        # Grab all the files in dirname, the selected directory
        files = os.listdir(dirname)

        # Loop through all files
        for f in files:

            # Join the name of the directory with the file name
            fullname = os.path.join(dirname, f)

            # If current element is a directory, add directory and recurse
            if os.path.isdir(fullname):
                treedata.Insert(parent, fullname, f, values=[], icon=folder_icon)
                add_files_in_folder(fullname, fullname)

            # If current element is a file, only add it if it ends in ".behavior"
            else:

                # Ensure that it is a ".behavior" file
                if f[-9:] == ".behavior":

                    # Check custom regions
                    df = pd.read_csv(fullname, header=2,sep="\t")
                    exp_custom_regions = list(df["Trigger Region Identifier"].dropna().unique())

                    # Adds all custom regions present in the directory to custom_regions_list
                    for region in exp_custom_regions:
                        if region not in custom_regions_list:
                            custom_regions_list.append(region)

                    # Insert treedata
                    treedata.Insert(parent, fullname, f, values=[], icon=file_icon)


    def delete_images():
        """
        delete_images(): Deletes all images in the Images directory, except for "Blank.png"
        Input:
            None
        Output:
            None
        """

        # Remove all images in Images directory
        list_path_delete = []

        # Iterate through all the images in the
        for img in os.listdir(path_to_images_folder):

            # Add everything except "Blank.png" to the list_path_delete that needs to be deleted
            if img != "Blank.png":
                list_path_delete.append(os.path.join(path_to_images_folder, img))

        # Delete everything within path_delete
        for img in list_path_delete:
            os.remove(img)


    def draw_figure(canvas, figure):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        # Problematic line
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)

        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def delete_fig(fig_to_delete):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        fig_to_delete.get_tk_widget().forget()
        plt.close('all')


    def get_plot_bounds(maze_type):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        # Ensure that the user provided a valid maze type
        maze_types = ['square', 'of', 'circle', 'y_maze', 'corridor']
        if maze_type.lower() not in maze_types:
            return print("Give a valid maze type")

        # Return appropriate plot bounds
        if maze_type.lower() == "circle":
            return [[-800, 800], [-800, 800]]
        elif maze_type.lower() == "corridor":
            return [[-100, 100], [-1400, 1400]]
        elif maze_type.lower() == "of":
            return [[-800, 800], [-800, 800]]
        elif maze_type.lower() == "square":
            return [[-800, 800], [-800, 800]],
        elif maze_type.lower() == "y_maze":
            return [[-600, 600], [-600, 600]]


    def place_grayscale_heatmap(x, y, maze_type, num_bin=default_num_bin_grayscale, title="", save=False, save_filepath=None, save_format=None):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        # Add boundaries - generalize for all boundaries
        key = str(maze_type).lower() + "_plotly"

        with open('maze_boundaries.json', 'r') as fp:
            all_maze_boundaries = json.load(fp)
            bound_x = all_maze_boundaries[key][0]
            bound_y = all_maze_boundaries[key][1]

        # Create the figure
        fig = plt.figure()
        ax = Subplot(fig, 111)
        fig.add_subplot(ax).hexbin(x, y, gridsize=num_bin, bins='log', cmap='binary')
        ax.plot(bound_x, bound_y)

        if maze_type.lower() == "y_maze":
            sep_x = [0, -125, 125, 0]
            sep_y = [-202.582, 13.925, 13.925, -202.582]
            ax.plot(sep_x, sep_y)

        plt.title(title)

        # Remove axes
        ax.axis["left"].set_visible(False)
        ax.axis["top"].set_visible(False)
        ax.axis["right"].set_visible(False)
        ax.axis["bottom"].set_visible(False)

        # Adjust figure size
        fig.set_size_inches(3, 3)

        if save:
            fig.savefig(save_filepath, format=save_format)

        return plt.gcf()


    def place_timescale_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_timescale, title="", save=False, save_filepath=None, save_format=None):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        # Make the indices
        x_indices = np.linspace(get_plot_bounds(maze_type)[0][0], get_plot_bounds(maze_type)[0][1], num_bin + 1)
        y_indices = np.linspace(get_plot_bounds(maze_type)[1][0], get_plot_bounds(maze_type)[1][1], num_bin + 1)

        # Make a num_bin X num_bin array of zeroes
        grid = np.zeros((num_bin, num_bin))

        # Loop through and add time diff to appropriate spot in grid
        for i in range(1, len(time_diff)):
            x_i = bisect(x_indices, X[i]) - 1
            y_i = abs(bisect(y_indices, Y[i]) - num_bin)
            grid[y_i][x_i] += time_diff[i]

        # Make subplots, grid, clear axis, and add title
        fig, ax = plt.subplots()
        im = ax.imshow(grid)
        ax.axis("off")
        plt.title(title)

        # Loop over data dimensions and create text annotations.
        size = 50 / num_bin
        for i in range(num_bin):
            for j in range(num_bin):
                val = round(grid[i, j], 1)
                if val >= 0.1:
                    if val >= 150:
                        text = ax.text(j, i, round(grid[i, j], 1), ha="center", va="center", color="black",
                                       size=size)
                    else:
                        text = ax.text(j, i, round(grid[i, j], 1), ha="center", va="center", color="w",
                                       size=size)

        # Adjust figure size
        fig.set_size_inches(3, 3)

        # Case where user wants to save the image
        if save:
            fig.savefig(save_filepath, format=save_format)

        return plt.gcf()


    def place_contour_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_contour, title="", save=False, save_filepath=None, save_format=None):
        """
        INSERT AWESOME DOCSTRING HERE
        Input:

        Output:

        """

        # Make the axis - depends on maze type
        x_indices = np.linspace(get_plot_bounds(maze_type)[0][0], get_plot_bounds(maze_type)[0][1], num_bin)
        y_indices = np.linspace(get_plot_bounds(maze_type)[1][0], get_plot_bounds(maze_type)[1][1], num_bin)

        # Make a num_bin X num_bin array of zeroes
        grid = np.zeros((num_bin, num_bin))

        # Loop through and add time diff to appropriate spot in grid
        for i in range(1, len(time_diff)):
            x_i = bisect(x_indices, X[i])
            y_i = bisect(y_indices, Y[i])
            grid[y_i][x_i] += time_diff[i]

        fig, ax = plt.subplots()

        # Add boundaries - generalize for all boundaries
        key = str(maze_type).lower() + "_plotly"

        with open('maze_boundaries.json', 'r') as fp:
            all_maze_boundaries = json.load(fp)
            bound_x = all_maze_boundaries[key][0]
            bound_y = all_maze_boundaries[key][1]


        plt.contourf(x_indices, y_indices, grid, 500, cmap="magma")
        ax.plot(bound_x, bound_y)

        plt.title(title)
        plt.axis("off")

        # bound_x, bound_y = updated_helper_functions.shapes(self.shape, plotly=True)
        bound_x = [-433.013, 0, 433.013, 558.013, 125, 125, -125, -125, -558.013, -433.013]
        bound_y = [-452.582, -202.582, -452.582, -236.075, 13.925, 513.924, 513.924, 13.925, -236.075, -452.582]

        sep_x = [0, -125, 125, 0]
        sep_y = [-202.582, 13.925, 13.925, -202.582]

        if maze_type.lower() == "y_maze":
            plt.plot(bound_x, bound_y)
            plt.plot(sep_x, sep_y)

        # Adjust figure size
        fig.set_size_inches(3, 3)

        # Case where user wants to save the image
        if save:
            fig.savefig(save_filepath, format=save_format)

        return plt.gcf()


    # Need to refresh
    window.Refresh()

    # Try opening the tree_information.json file
    try:
        with open("tree_information.json", "r") as f:

            # Try loading the tree with the file provided in tree_information. Catch ValueError if file is empty
            try:

                # Load the data and pass it into add_files_in_folder function
                json_data = json.load(f)
                treedata = sg.TreeData()
                print(json_data["tree"])
                add_files_in_folder('', json_data["tree"])

                # Update tree and refresh window
                window["-TREE-"].update(values=treedata)
                window.Refresh()

            # Catch error if tree_information.json is empty
            except ValueError:
                print("Empty File!")

    # Catch error of incorrect name of json file
    except FileNotFoundError:
        print("Can not open file")



    # ===================================================== LOOP ===================================================== #

    try:
        # Event Loop to process events
        while True:

            # Read in the event and values
            event, values = window.read()

            print('\n')

            print("Event:", event)
            print("Value:", values)

            # Display the index of the current experiment if there is one
            if index_current_experiment is not None:
                print("index_current_experiment:", index_current_experiment)
            else:
                print("index_current_experiment is of NoneType")


            # If there is a selected experiment, grab the first/only element inside
            # THIS IS COMING UP WITH ERROR BECAUSE NONETYPE IS NOT SUBSCRIPTABLE BUT WHEN DOES TABLE BECOME NONE?
            # PERHAPS IT HAS TO DO WITH INDEX_CURRENT_EXPERIMENT BEING SET TO NONE?
            # NEED TO FIGURE THIS OUT

            # If there are values and there is a value for table
            if values is not None and len(values['-TABLE-']) > 0:

                # Get the first value for table, the index for the current user-selected experiment
                index_current_experiment = values['-TABLE-'][0]


                # BEGIN EXPERIMENTAL REGION

                # Grab the index of selected experiment and of "file_name" column in summary_stats_master_df
                index_file_name = summary_stats_master_df.columns.get_loc("file_name")
                title_selected_experiment = summary_stats_master_df.iloc[index_current_experiment, index_file_name]
                df = all_exp_dfs_dict[title_selected_experiment]

                X = df["Position.X"]
                Y = df["Position.Y"]
                time_diff = df["#Snapshot Timestamp"].diff()
                maze_type = df["maze_type"][0]

                # END EXPERIMENTAL REGION












            # Attempt to load in the default save path
            try:
                # If the user has saved something somewhere before, grab that location
                with open("default_save_location.json", "r") as file:
                    default_save_path = json.load(file)

            # If there is nothing in the JSON, set the default save path to None
            except json.decoder.JSONDecodeError:
                default_save_path = None








            # Open
            if event == "Open":

                # Delete everything in custom_regions_list
                custom_regions_list = []

                # Should probably add some check here that ensures that maze types are not duplicated with same name
                print("Open Button Pushed ... starting to update tree")

                # Get initial path
                try:
                    with open("tree_information.json", "r") as f:
                        initial_path = json.load(f)

                        # Obtain Starting Path
                        if os.path.isdir(initial_path["tree"]):
                            STARTING_PATH = sg.PopupGetFolder('Folder to analyze',
                                                              initial_folder=initial_path,
                                                              no_window=True)
                        else:
                            STARTING_PATH = sg.PopupGetFolder('Folder to analyze',
                                                              no_window=True)

                except FileNotFoundError:
                    print("File not found")

                # Get Tree Data
                treedata = sg.TreeData()

                try:
                    add_files_in_folder('', STARTING_PATH)

                    # Update Tree
                    window["-TREE-"].update(values=treedata)
                    window.Refresh()

                    # Update json File to keep location of file starting path
                    print(STARTING_PATH)

                    # Update dictionary in JSON
                    print(custom_regions_list)
                    with open(tree_filename, 'w') as f_obj:
                        dummy_dictionary = {
                            "tree": STARTING_PATH,
                            "appropriate_custom_regions": custom_regions_list
                        }
                        json.dump(dummy_dictionary, f_obj)

                except FileNotFoundError:
                    continue

                # Restart program
                delete_images()
                window.close()
                main()

            # Process
            elif event == "Process":

                print("Beginning of Process:", values['-TABLE-'])

                print("Process Button Pushed ... updating table")
                for exp in values['-TREE-']:

                    print('\n')

                    try:
                        # exp is a filepath to a behavior path
                        print("EXP:", exp)

                        # If exp is a directory, raise an error to skip adding to the table
                        if os.path.isdir(exp):
                            print("Directory detected")
                            raise ValueError("Directories are not accepted")

                        # With exp, make MouseExperiment object called mouse
                        mouse = classes.MouseExperiment(exp)

                        # Get mouse.metadata (dictionary) and make a dataframe out of it
                        exp_metadata = mouse.metadata

                        headers_list = list(headers.keys())
                        exp_metadata_headers_list = list(exp_metadata.keys())
                        exp_metadata_values_list = list(exp_metadata.values())

                        # We can insert into exp_metadata
                        for header in headers_list:
                            if header not in exp_metadata_headers_list:

                                print(header, "not in list")

                                index_header = headers_list.index(header)

                                exp_metadata_headers_list.insert(index_header, header)
                                exp_metadata_values_list.insert(index_header, "None")

                        # Zip exp_metadata lists into dictionary
                        exp_metadata = dict(zip(exp_metadata_headers_list, exp_metadata_values_list))
                        exp_metadata_df = pd.DataFrame([exp_metadata])

                        # Raise an error and skip if experiment metadata values are already in the table
                        if list(exp_metadata.values()) in tableValues:
                            print("Duplicate data detected - be careful")
                            raise ValueError("Duplicates are not accepted")

                        # If not duplicate data, update summary_stats_master_df, all_exp_dfs_dict, and table
                        # concatenate data with summary_stats_master_df
                        summary_stats_master_df = pd.concat([summary_stats_master_df, exp_metadata_df],
                                                            ignore_index=True)

                        # Add raw data to all_exp_dfs_dict
                        all_exp_dfs_dict[mouse.title] = mouse.df

                        # Update tableValues with exp.metadata
                        tableValues.append(list(exp_metadata.values()))

                        # Update table and refresh window
                        window["-TABLE-"].update(values=tableValues)
                        window.Refresh()

                        # Load experiment images to Images folder
                        try:
                            base = os.path.dirname(os.path.abspath("Images/Blank.png"))

                            print("base:", base)
                            where_to_save = os.path.join(base, mouse.title)
                            print("where_to_save:", where_to_save)

                            # Construct Movement Plots, heatmaps, speed plot, center_anxiety, and place in Images folder
                            mouse.movement_plot(save=True, save_filepath=where_to_save)
                            mouse.heatmap(save=True, save_filepath=where_to_save)
                            mouse.movement_plot(region="center_anxiety", save=True, save_filepath=where_to_save)
                            mouse.wall_distance_plot(save=True, save_filepath=where_to_save)
                            mouse.speed_plot(save=True, save_filepath=where_to_save, average=True, median=True)

                        except ValueError:
                            print("Not saving the image to folder")
                            continue

                    # Handle errors by skipping those values of exp
                    except ValueError:
                        print("Skipping", exp)
                        continue

                # Print statement. Find out how long columns list is
                print("length of summary stats df", len(summary_stats_master_df.columns))

                print("Ending of Process:", values['-TABLE-'])

            # Plot
            elif event == "-PLOT-":

                print("Within PLOT:", values['-TABLE-'])

                if values["-PLOT-"] == "Movement Plot":
                    plot_type = "_movement_plot"
                elif values["-PLOT-"] == "Heatmap":
                    plot_type = "_heatmap"
                elif values["-PLOT-"] == "Center Anxiety":
                    plot_type = "_center_anxiety"
                elif values["-PLOT-"] == "Speed":
                    plot_type = "_speed"
                elif values["-PLOT-"] == "Wall Distance":
                    plot_type = "_wall_distance"
                elif values["-PLOT-"] == "Custom Regions":
                    plot_type = "_custom_regions"

                print("updated plot_type to", plot_type)

                # If there is a row selected in table, update the canvas
                if values['-TABLE-']:
                    print("There is a row selected")

                    title_selected_experiment = summary_stats_master_df.iloc[index_current_experiment, index_file_name]

                    image_name = "Images/" + title_selected_experiment[:-9] + plot_type + "_sized.png"
                    window['-IMAGE-PLOT-'].update(filename=image_name, visible=True)

                    # Refresh statements
                    window.Refresh()

                    print("Finished updating canvas with image")

                else:
                    print("No row selected")

            # Click on table row to get plot
            elif event == "-TABLE-":

                print("Within TABLE:", values['-TABLE-'])

                # Grab the index of the selected experiment and of "file_name" column in the summary_stats_master_df
                index_file_name = summary_stats_master_df.columns.get_loc("file_name")
                title_selected_experiment = summary_stats_master_df.iloc[index_current_experiment, index_file_name]
                image_name = "Images/" + title_selected_experiment[:-9] + plot_type + "_sized.png"

                # Update the plot
                window['-IMAGE-PLOT-'].update(filename=image_name, visible=True)

                # Update Processed Data Table. First Grab and slice the first 100 lines of processed df
                df = all_exp_dfs_dict[title_selected_experiment]
                df_sliced = df.iloc[:1000, :]
                window["-PROCESSED-TABLE-"].update(values=df_sliced.values.tolist())

                # Update the filename text
                window["-ENHANCED-BEHAVIORAL-DISPLAY-"].update(f"Preview of: {title_selected_experiment}")




                # UPDATE ALL THREE HEATMAPS

                # Obtain the X and Y coordinates of currently selected experiment
                X = df["Position.X"]
                Y = df["Position.Y"]
                time_diff = df["#Snapshot Timestamp"].diff()

                # Determine the Maze Type
                maze_type = df["maze_type"][0]






                # UPDATE GRAYSCALE HEATMAP

                # Clear current grayscale image
                if fig_grayscale is not None:
                    delete_fig(fig_grayscale)

                # Get new grayscale image
                if values['-SLIDER-GRAYSCALE-'] and values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:
                    fig = place_grayscale_heatmap(X, Y, maze_type, num_bin=int(values['-SLIDER-GRAYSCALE-']), title=title_selected_experiment[:-9])
                elif values["-SLIDER-GRAYSCALE-"]:
                    fig = place_grayscale_heatmap(X, Y, maze_type, num_bin=int(values['-SLIDER-GRAYSCALE-']))
                elif values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:
                    fig = place_grayscale_heatmap(X, Y, maze_type, num_bin=default_num_bin_grayscale, title=title_selected_experiment[:-9])
                else:
                    fig = place_grayscale_heatmap(X, Y, maze_type, num_bin=default_num_bin_grayscale)

                # Update window with new grayscale fig
                fig_grayscale = draw_figure(window['-INTERACTIVE-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-HEATMAP-"].update()
                window.Refresh()




                # UPDATE TIMESCALE HEATMAP

                # Clear current timescale heatmap
                if fig_timescale is not None:
                    delete_fig(fig_timescale)

                # Get new timescale fig
                if values['-SLIDER-TIMESCALE-'] and values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:
                    new_fig_timescale = place_timescale_heatmap(X, Y, time_diff, maze_type, num_bin=int(values['-SLIDER-TIMESCALE-']), title=title_selected_experiment[:-9])
                elif values["-SLIDER-TIMESCALE-"]:
                    new_fig_timescale = place_timescale_heatmap(X, Y, time_diff, maze_type, num_bin=int(values['-SLIDER-TIMESCALE-']))
                elif values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:
                    new_fig_timescale = place_timescale_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_timescale, title=title_selected_experiment[:-9])
                else:
                    new_fig_timescale = place_timescale_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_timescale)

                # Update window with new timescale fig
                fig_timescale = draw_figure(window['-INTERACTIVE-TIMESCALE-HEATMAP-'].TKCanvas, new_fig_timescale)
                window["-INTERACTIVE-TIMESCALE-HEATMAP-"].update()
                window.Refresh()






                # UPDATE CONTOUR HEATMAP

                # Clear current contour heatmap
                if fig_contour is not None:
                    delete_fig(fig_contour)

                # Get new contour fig
                if values['-SLIDER-CONTOUR-'] and values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                    new_fig_contour = place_contour_heatmap(X, Y, time_diff, maze_type, num_bin=int(values['-SLIDER-CONTOUR-']), title=title_selected_experiment[:-9])
                elif values["-SLIDER-CONTOUR-"]:
                    new_fig_contour = place_contour_heatmap(X, Y, time_diff, maze_type, num_bin=int(values['-SLIDER-CONTOUR-']))
                elif values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                    new_fig_contour = place_contour_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_contour, title=title_selected_experiment[:-9])
                else:
                    new_fig_contour = place_contour_heatmap(X, Y, time_diff, maze_type, num_bin=default_num_bin_contour)

                # Update window with new timescale fig
                fig_contour = draw_figure(window['-INTERACTIVE-CONTOUR-HEATMAP-'].TKCanvas, new_fig_contour)
                window["-INTERACTIVE-CONTOUR-HEATMAP-"].update()
                window.Refresh()






            # I ran into a strange error with this part of the code. i plotted several exps, including y maze and saved a csv
            # then I tried saving another, and it terminated the program.
            elif event == "-SAVE-METADATA-":

                print("-SAVE-METADATA- clicked")

                # Get the directory and document name
                path = sg.PopupGetFolder('Select directory to save in', no_window=True)
                if path != "":
                    name = sg.PopupGetText('Name of document')
                else:
                    name = ""

                # Ensure that .csv is at the end
                if name is not None and name[-4:] != ".csv":
                    name += ".csv"

                # Ensure that path and name are strings and not empty
                if isinstance(path, str) and isinstance(name, str) and path != "" and name != "":
                    # Join the directory and document names together
                    filepath = os.path.join(path, name)

                    # Save the summary_stats_master_df to the desired filepath, and strip the index off
                    summary_stats_master_df.to_csv(filepath, sep='\t', index=False)

            elif event == "-COPY-METADATA-":
                print("-COPY-METADATA- clicked")

                # copy the DataFrame to the system clipboard
                summary_stats_master_df.to_clipboard(index=False)

            elif event == "-SAVE-IMG-":
                # Grab the image type
                img_type = values["-SAVE-IMG-"]

                # Verification statements
                print("-SAVE-IMG- clicked")
                print("img_type:", img_type)
                print("plot_type:", plot_type)

                # Ensure that there is an image to display
                if values["-SAVE-IMG-"] and values["-TABLE-"] and img_type and plot_type:
                    print("conditions to save image are met")

                    # Get index of selected row, then get the appropriate file_path
                    index = index_current_experiment
                    exp_filepath = tableValues[index][1]

                    # Set default name by taking last item in file_path after \ and adding the plot_type and img_type
                    default_name = os.path.basename(os.path.normpath(exp_filepath))[:-9]
                    default_name += plot_type
                    default_name += img_type

                    print(default_name)

                    # Get the directory and document name
                    path = sg.PopupGetFolder('Select directory to save in', no_window=True, )
                    if path != "":
                        name = sg.PopupGetText('Name of image', default_text=default_name)
                    else:
                        name = ""

                    # Ensure that path and name are strings and not empty
                    if isinstance(path, str) and isinstance(name, str) and path != "" and name != "":

                        # Add correct extension
                        if img_type not in name:
                            name += img_type

                        # Join the directory and document names together
                        img_filepath = os.path.join(path, name)

                        print("Will save to", img_filepath)

                        # Get rid of period so you can use it as img_type parameter of savefig
                        extension = img_type[1:]

                        shutil.copy(os.path.join(path_to_images_folder, default_name), img_filepath)

                        print("Should have saved")

                    else:
                        print("TRANSACTION NOT COMPLETED")

                else:
                    print("Conditions to save image are NOT met")

            # SAVE RAW DF
            elif event == '-SAVE-PROCESSED-DATA-':
                print("Saving  df...")

                # Ensure that an experiment is selected
                if values['-TABLE-']:

                    # Grab the index of the selected experiment and of "file_name" column in the summary_stats_master_df
                    index_selected_experiment = index_current_experiment
                    index_file_name = summary_stats_master_df.columns.get_loc("file_name")

                    # Using these indices, obtain the title of the selected experiment
                    title_selected_experiment = summary_stats_master_df.iloc[index_selected_experiment, index_file_name]

                    # Grab the index of selected experiment, and then the filepath of the experiment
                    print(title_selected_experiment)

                    # Get the directory and document name
                    default_title = title_selected_experiment[:-9] + "_enhanced_behavior.csv"
                    path = sg.PopupGetFolder('Select directory to save in', no_window=True, )
                    if path != "":
                        name = sg.PopupGetText('Name of document', default_text=default_title)
                    else:
                        name = ""

                    # Ensure that path and name are strings and not empty
                    if isinstance(path, str) and isinstance(name, str) and path != "" and name != "":
                        print("Conditions met to save")

                        # Ensure that ".csv" is last in the name, so it saves and opens correctly
                        if ".csv" not in name[-4:]:
                            name += ".csv"

                        # Join the directory and document names together
                        filepath = os.path.join(path, name)

                        # Grab the appropriate df
                        df_to_save = all_exp_dfs_dict[title_selected_experiment]
                        df_to_save.to_csv(filepath, sep='\t', index=False)
                        print("enhanced behavior csv saved")

                    else:
                        print("Empty path and/or name")

                else:
                    print("Empty Table")


            # =============================================== GRAYSCALE ============================================== #


            # GRAYSCALE TITLE
            elif event == "-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-":

                # Ensure that there is a selected experiment
                if index_current_experiment is None:
                    continue

                # Clear current fig_grayscale
                if fig_grayscale is not None:
                    delete_fig(fig_grayscale)

                # If user indicates to add a title
                if values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:

                    # Get new fig with title
                    fig = place_grayscale_heatmap(X,
                                                  Y,
                                                  maze_type,
                                                  int(values['-SLIDER-GRAYSCALE-']),
                                                  title_selected_experiment[:-9])

                # If user indicates to remove a title
                else:

                    # Get new fig without title
                    fig = place_grayscale_heatmap(X,
                                                  Y,
                                                  maze_type,
                                                  int(values['-SLIDER-GRAYSCALE-']))

                # Update window with new grayscale
                fig_grayscale = draw_figure(window['-INTERACTIVE-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-HEATMAP-"].update()
                window.Refresh()

            # GRAYSCALE SLIDER
            elif event == "-SLIDER-GRAYSCALE-":

                # Ensure that there
                if fig_grayscale is None:
                    continue

                # Clear current grayscale image
                if fig_grayscale is not None:
                    delete_fig(fig_grayscale)

                # Make new grayscales with new slider value

                # If user wants title
                if values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:
                    fig = place_grayscale_heatmap(X,
                                                  Y,
                                                  maze_type,
                                                  int(values['-SLIDER-GRAYSCALE-']),
                                                  title_selected_experiment[:-9])

                # If user does not want a title
                else:
                    fig = place_grayscale_heatmap(X,
                                                  Y,
                                                  maze_type,
                                                  int(values['-SLIDER-GRAYSCALE-']))

                # Update window
                fig_grayscale = draw_figure(window['-INTERACTIVE-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-HEATMAP-"].update()
                window.Refresh()

            # GRAYSCALE SAVE
            elif event == '-SAVE-GRAYSCALE-HEATMAP-':

                # Grab the desired image format
                extension = values['-SAVE-GRAYSCALE-HEATMAP-']

                # Ensure that there is an image to replicate
                if fig_grayscale is None:
                    continue

                # Grab the directory to save the image in
                if default_save_path:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             initial_folder=default_save_path,
                                             no_window=True)
                else:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             no_window=True)

                # Ensure a valid directory is given
                if path is not None and path != "":

                    # Write the path value into json
                    with open("default_save_location.json", 'w') as file:
                        json.dump(path, file)

                    # Grab the name of the experiment
                    name = sg.PopupGetText('Name of document',
                                           default_text=str(title_selected_experiment[:-9] +
                                                            "_grayscale_" +
                                                            str(int(values['-SLIDER-GRAYSCALE-']))))

                    # Ensure a valid name is given
                    if name is not None and name != "":

                        # Add proper extension if necessary
                        if name[-4:] != extension:
                            name += extension

                        # Join the directory and document names together
                        filepath = os.path.join(path, name)

                        # Create and save new grayscale with slider value and title
                        if values['-SLIDER-GRAYSCALE-'] and values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:
                            place_grayscale_heatmap(X,
                                                    Y,
                                                    maze_type,
                                                    num_bin=int(values['-SLIDER-GRAYSCALE-']),
                                                    title=title_selected_experiment[:-9],
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save new grayscale with slider value, and no title
                        elif values["-SLIDER-GRAYSCALE-"]:
                            place_grayscale_heatmap(X,
                                                    Y,
                                                    maze_type,
                                                    num_bin=int(values['-SLIDER-GRAYSCALE-']),
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save new grayscale with default grayscale num bin value and title
                        elif values["-CHECKBOX-TITLE-GRAYSCALE-HEATMAP-"]:
                            place_grayscale_heatmap(X,
                                                    Y,
                                                    maze_type,
                                                    num_bin=default_num_bin_grayscale,
                                                    title=title_selected_experiment[:-9],
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save new grayscale with default grayscale num bin value and no title
                        else:
                            place_grayscale_heatmap(X,
                                                    Y,
                                                    maze_type,
                                                    num_bin=default_num_bin_grayscale,
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])


            # =============================================== TIMESCALE ============================================== #


            # TIMESCALE TITLE
            elif event == "-CHECKBOX-TITLE-TIMESCALE-HEATMAP-":

                # Ensure that there is a selected experiment
                if index_current_experiment is None:
                    continue

                # Clear current fig_timescale
                if fig_timescale is not None:
                    delete_fig(fig_timescale)

                # If user indicates to add a title
                if values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:

                    # Get new fig with title
                    fig = place_timescale_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  int(values['-SLIDER-TIMESCALE-']),
                                                  title_selected_experiment[:-9])

                # If user indicates to remove a title
                else:

                    # Get new fig without title
                    fig = place_timescale_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  int(values['-SLIDER-TIMESCALE-']))

                # Update window with fig
                fig_timescale = draw_figure(window['-INTERACTIVE-TIMESCALE-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-TIMESCALE-HEATMAP-"].update()
                window.Refresh()

            # TIMESCALE SLIDER
            elif event == "-SLIDER-TIMESCALE-":

                # Ensure that there is already a timescale to act on
                if fig_timescale is None:
                    continue

                # Clear current grayscale image
                if fig_timescale is not None:
                    delete_fig(fig_timescale)

                # Make new timescales with new slider value

                # If user wants title
                if values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:
                    fig = place_timescale_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  int(values['-SLIDER-TIMESCALE-']),
                                                  title_selected_experiment[:-9])

                # If user does not want title
                else:
                    fig = place_timescale_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  int(values['-SLIDER-TIMESCALE-']))

                # Update window
                fig_timescale = draw_figure(window['-INTERACTIVE-TIMESCALE-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-TIMESCALE-HEATMAP-"].update()
                window.Refresh()

            # TIMESCALE SAVE
            elif event == '-SAVE-TIMESCALE-HEATMAP-':

                # Grab the desired image format
                extension = values['-SAVE-TIMESCALE-HEATMAP-']

                # Ensure that there is an image to replicate
                if fig_timescale is None:
                    continue

                # Grab the directory to save the image in
                if default_save_path:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             initial_folder=default_save_path,
                                             no_window=True)
                else:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             no_window=True)

                # Ensure a valid directory is given
                if path is not None and path != "":

                    # Write the path value into json
                    with open("default_save_location.json", 'w') as file:
                        json.dump(path, file)

                    # Grab the name of the experiment
                    name = sg.PopupGetText('Name of document',
                                           default_text=str(title_selected_experiment[:-9] +
                                                            "_timescale_" +
                                                            str(int(values['-SLIDER-TIMESCALE-']))))

                    # Ensure a valid name is given
                    if name is not None and name != "":

                        # Add proper extension if necessary
                        if name[-4:] != extension:
                            name += extension

                        # Join the directory and document names together
                        filepath = os.path.join(path, name)

                        # Create and save timescale with both slider value and title
                        if values['-SLIDER-TIMESCALE-'] and values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:
                            place_timescale_heatmap(X,
                                                    Y,
                                                    time_diff,
                                                    maze_type,
                                                    num_bin=int(values['-SLIDER-TIMESCALE-']),
                                                    title=title_selected_experiment[:-9],
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save timescale with slider value, and no title
                        elif values["-SLIDER-TIMESCALE-"]:
                            place_timescale_heatmap(X,
                                                    Y,
                                                    time_diff,
                                                    maze_type,
                                                    num_bin=int(values['-SLIDER-TIMESCALE-']),
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save timescale with default timescale num bin, and title
                        elif values["-CHECKBOX-TITLE-TIMESCALE-HEATMAP-"]:
                            place_timescale_heatmap(X,
                                                    Y,
                                                    time_diff,
                                                    maze_type,
                                                    num_bin=default_num_bin_timescale,
                                                    title=title_selected_experiment[:-9],
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])

                        # Create and save timescale with default timescale num bin, and no title
                        else:
                            place_timescale_heatmap(X,
                                                    Y,
                                                    time_diff,
                                                    maze_type,
                                                    num_bin=default_num_bin_timescale,
                                                    save=True,
                                                    save_filepath=filepath,
                                                    save_format=img_types[extension])


            # ================================================ CONTOUR =============================================== #


            # CONTOUR TITLE
            elif event == "-CHECKBOX-TITLE-CONTOUR-HEATMAP-":

                # Ensure that there is a selected experiment
                if index_current_experiment is None:
                    continue

                # Clear current contour heatmap
                if fig_contour is not None:
                    delete_fig(fig_contour)

                # If user indicates to add title
                if values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                    fig = place_contour_heatmap(X,
                                                Y,
                                                time_diff,
                                                maze_type,
                                                int(values['-SLIDER-CONTOUR-']),
                                                title_selected_experiment[:-9])

                # If user indicates to remove title
                else:
                    fig = place_contour_heatmap(X,
                                                Y,
                                                time_diff,
                                                maze_type,
                                                int(values['-SLIDER-CONTOUR-']))

                # Update window with new contour heatmap
                fig_contour = draw_figure(window['-INTERACTIVE-CONTOUR-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-CONTOUR-HEATMAP-"].update()
                window.Refresh()

            # CONTOUR SLIDER
            elif event == "-SLIDER-CONTOUR-":

                # Ensure there is already a contour heatmap to act on
                if fig_contour is None:
                    continue

                # Clear current contour heatmap
                if fig_contour is not None:
                    delete_fig(fig_contour)

                # Make new contour heatmap with title and slider value
                if values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                    fig = place_contour_heatmap(X,
                                                Y,
                                                time_diff,
                                                maze_type,
                                                int(values['-SLIDER-CONTOUR-']),
                                                title_selected_experiment[:-9])

                # Make new contour heatmap with no title and slider value
                else:
                    fig = place_contour_heatmap(X,
                                                Y,
                                                time_diff,
                                                maze_type,
                                                int(values['-SLIDER-CONTOUR-']))

                # Update window
                fig_contour = draw_figure(window['-INTERACTIVE-CONTOUR-HEATMAP-'].TKCanvas, fig)
                window["-INTERACTIVE-CONTOUR-HEATMAP-"].update()
                window.Refresh()

            # CONTOUR SAVE
            elif event == '-SAVE-CONTOUR-HEATMAP-':

                # Grab the desired image format
                extension = values['-SAVE-CONTOUR-HEATMAP-']

                # Ensure that there is an image to replicate
                if fig_contour is None:
                    continue

                # Grab the directory to save the image in
                if default_save_path:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             initial_folder=default_save_path,
                                             no_window=True)
                else:
                    path = sg.PopupGetFolder('Select directory to save in',
                                             no_window=True)

                # Ensure a valid directory is given
                if path is not None and path != "":

                    # Write the path value into json
                    with open("default_save_location.json", 'w') as file:
                        json.dump(path, file)

                    # Grab the name of the experiment
                    name = sg.PopupGetText('Name of document',
                                           default_text=str(title_selected_experiment[:-9] +
                                                            "_contour_" +
                                                            str(int(values['-SLIDER-CONTOUR-']))))

                    # Ensure a valid name is given
                    if name is not None and name != "":

                        # Add proper extension if necessary
                        if name[-4:] != extension:
                            name += extension

                        # Join the directory and document names together
                        filepath = os.path.join(path, name)

                        # Make and save new contour heatmap with slider value and title
                        if values['-SLIDER-CONTOUR-'] and values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                            place_contour_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  num_bin=int(values['-SLIDER-CONTOUR-']),
                                                  title=title_selected_experiment[:-9],
                                                  save=True,
                                                  save_filepath=filepath,
                                                  save_format=img_types[extension])

                        # Make and save new contour heatmap with slider value, and no title
                        elif values["-SLIDER-CONTOUR-"]:
                            place_contour_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  num_bin=int(values['-SLIDER-CONTOUR-']),
                                                  save=True,
                                                  save_filepath=filepath,
                                                  save_format=img_types[extension])

                        # Make and save new contour heatmap with default contour num bin and title
                        elif values["-CHECKBOX-TITLE-CONTOUR-HEATMAP-"]:
                            place_contour_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  num_bin=default_num_bin_contour,
                                                  title=title_selected_experiment[:-9],
                                                  save=True,
                                                  save_filepath=filepath,
                                                  save_format=img_types[extension])

                        # Make and save new contour heatmap with default contour num bin and no title
                        else:
                            place_contour_heatmap(X,
                                                  Y,
                                                  time_diff,
                                                  maze_type,
                                                  num_bin=default_num_bin_contour,
                                                  save=True,
                                                  save_filepath=filepath,
                                                  save_format=img_types[extension])


            # CLEAR ALL
            elif event == "Clear All":

                # Delete all images in Images directory
                delete_images()

                # Close and restart program
                window.close()
                main()

            # REMOVE
            elif event == "Remove":

                # I noticed a bug that needs resolving
                # When you remove an experiment and then immediately shift between different plot types.
                # Doesn't crash, but should be address

                # Ensure that an experiment is selected
                if values['-TABLE-']:

                    # Verification statements
                    print("index_current_experiment", index_current_experiment)
                    print("trying to remove values[-TABLE-][0]", values["-TABLE-"][0])
                    print("tableValues", tableValues)
                    print("length of tableValues", len(tableValues))

                    # Clear the item from metadata summary
                    summary_stats_master_df = summary_stats_master_df.drop([values["-TABLE-"][0]]).reset_index(drop=True)

                    # Clear from table function
                    def clear_from_table():
                        del tableValues[values["-TABLE-"][0]]
                        window["-TABLE-"].update(values=tableValues)
                        window.Refresh()

                    # Removing the first item in the table
                    if values["-TABLE-"][0] == 0:

                        # There are other items in the table, other than the first item
                        if len(tableValues) > 1:
                            print("removing first, there are additional")
                            clear_from_table()

                        # There are no other items in the table
                        else:
                            print("removing the only item in the table")
                            window.close()
                            main()

                    # Not removing the first item in the table
                    else:

                        # Removing a middle item in the table (not first, not last)
                        if values["-TABLE-"][0] < len(tableValues) - 1:
                            print("removing a middle item")
                            clear_from_table()

                        # Removing last item in the table
                        else:
                            print("Removing the last item in the table")
                            clear_from_table()

                            # Shift index down
                            values['-TABLE-'][0] -= 1
                            index_current_experiment -= 1

            # EXIT
            elif event == "Exit":

                # Clear all images in Images directory
                delete_images()

                # Break out of loop
                break

            # CLOSE WINDOW
            elif event == sg.WIN_CLOSED:

                # Clear all images in Images directory
                delete_images()

                # Break out of loop
                break

    # DELETE ALL IMAGES
    finally:
        delete_images()

    # Close window
    window.close()

# MAIN LOOP
if __name__ == "__main__":
    main()
