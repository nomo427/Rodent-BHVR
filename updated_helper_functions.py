from shapely.geometry import Point
from shapely.geometry import Polygon, LineString, LinearRing, MultiPoint
import math
import numpy as np
import pandas as pd
import time
import os


def behavioral_to_df(behavioral_file_path):
    """
    behavioral_to_df(): Turns behavior file into dataframe. Ignores metadata of first two rows.
    Input:
        behavior_file_path (string):                 File path of behavior file on your computer
    Output:
        df (Pandas Dataframe):                       Pandas dataframe of behavioral file
    """

    # Time the function
    start = time.time()

    # Read the tsv into a pandas dataframe
    df = pd.read_csv(behavioral_file_path, header=2, sep='\t')

    # End the time of the function
    print("FUNC behavioral_file_path took", str(time.time() - start))

    # Return the pandas dataframe
    return df


def calc_speed(behavior_df):
    """
    calc_speed(): Calculates and returns instantaneous speeds of subject
    Input:
        behavior_df (Pandas Dataframe):             Behavior Dataframe
    Output:
        df["Speed"] (Column of Pandas Dataframe):   Contains all instantaneous speeds in dataframe column format
    """

    # Time the function
    start = time.time()

    # Copy the Dataframe to ensure no changes are made to original
    df = behavior_df.copy()

    # Create columns of differences in x position, y position, and time difference
    df["X_diff"] = df["Position.X"].diff()
    df["Y_diff"] = df["Position.Y"].diff()
    df["Time_diff"] = df["#Snapshot Timestamp"].diff()

    # List of speeds
    speeds = [0.0,]

    # Iterate through the columns, starting with 2nd row (row index 1)
    df_temp = df.iloc[1:,:]
    for row_name, row in df_temp.iterrows():
        a = row["X_diff"]
        b = row["Y_diff"]
        t = row["Time_diff"]

        # Find c using c = square root of sum of a_squared and b_squared
        c = math.sqrt(math.pow(a, 2) + math.pow(b, 2))

        # Try to calculate speed by dividing distance by time
        try:
            speed = c / t
            speeds.append(speed)

        # If time is 0, just append 0 to speeds
        except ZeroDivisionError:
            speed = 0
            speeds.append(speed)

    # Add column to dataframe
    df["Speed"] = speeds

    # End time of function
    print("FUNC calc_speed took", str(time.time() - start))

    # Return the speeds column
    return df["Speed"]


def shapes(maze_array, plotly=False):
    """
    shapes(): Organizes maze coordinates for function use.
        Basically a fancy dictionary. Input the maze type to get the coordinates
        This will definitely need to change with updates from Ryan
    Input:
        maze_array (string):        Options are 'square', 'OF', 'circle', 'y_maze', 'corridor'
        plotly (bool):              Default set to False. If True, extra coordinates get appended, to close the shape
    Output:
        Coords:                     Coordinates for corresponding maze type
    """

    # Time the function
    start = time.time()

    # Ensure that the user provided a valid maze type
    maze_types = ['square', 'of', 'circle', 'y_maze', 'corridor']
    if maze_array.lower() not in maze_types:
        return print("Give a valid maze type")

    # Get coordinates for square and open field mazes
    if (maze_array.lower() == 'square') or (maze_array.lower() == 'of'):
        coords = np.array([[750, -750],
                           [-750, -750],
                           [-750, 750],
                           [750, 750]])

        if plotly:
            x, y = coords.T
            x = np.append(x, [750])
            y = np.append(y, [-750])
            return x, y

        else:

            # Return coordinates
            return coords

    # Get radius for circle maze
    elif maze_array.lower() == 'circle':

        # This coord is the radius
        coords = 750

        # Return coordinates
        return coords

    # Get coordinates for Y maze
    elif maze_array.lower() == 'y_maze':
        coords = np.array([[-433.013, -452.582],
                               [0, -202.582],
                               [433.013, -452.582],
                               [558.013, -236.075],
                               [125, 13.925],
                               [125, 513.924],
                               [-125, 513.924],
                               [-125, 13.925],
                               [-558.013, -236.075]])

        # If need to make a plotly outline, append the first coordinates to close boundaries
        if plotly:

            # Transpose and add appropriate x and y values
            x, y = coords.T
            x = np.append(x, [-433.013])
            y = np.append(y, [-452.582])

            # Return
            return x, y

        else:
            return coords

    # Get coordinates for corridor maze
    elif maze_array.lower() == 'corridor':
        coords = np.array([[-42, -1332.286],
                           [-42, 1332.286],
                           [42, 1332.286],
                           [42, -1332.286]])

        # If need to make a plotly outline, append the first coordinates to close boundaries
        if plotly:

            # Transpose and add appropriate x and y values
            x, y = coords.T
            x = np.append(x, -42)
            y = np.append(y, -1332.286)

            # Return
            return x, y

        else:
            return coords

    # Display how long function took
    print("FUNC shapes took", str(time.time() - start))


def mouse_edge_distance(behavior_df, maze_type, distance_threshold):
    """
    mouse_edge_distance(): Detects when mouse is near boundaries of maze.
    Input:
        behavior_df (Pandas Dataframe):     Pandas behavior dataframe
        maze_type (string):                 String label of maze. Options are [square, OF, corridor, y_maze, circle]
        distance_threshold (real number):   The percent distance from wall of mouse. Must be between 0 and 1
    Output:
        dist_wall_units (Column of Pandas Dataframe):        Unit distance from wall for each timestamp
        distance_marked (Column of Pandas Dataframe):        Column with 1 if within threshold, 0 otherwise
    """

    # Time the function
    start = time.time()

    # Ensure that distance_threshold is between 0 and 1
    if distance_threshold >= 1 or distance_threshold < 0:
        return print("Input a distance threshold between 0 and 1.")

    # Create function
    def shape_shrink(maze_polygon, distance_threshold):
        """
        shape_shrink(): Shrinks the maze polygon down by multiplying the maze coordinates by desired distance threshold
        Input:
            maze_polygon (np.ndarray or integer):       The array of maze coordinates (must be in plotly form)
            distance_threshold (real number):           Percent distance from wall of mouse
        Output:
            polygon_maze (Polygon shape):               Polygon form of original maze
            reduced_poly (Polygon shape):               Polygon form of both original maze AND shrunken maze
        """

        # Subtract threshold from one so you know what to multiply maze polygon by
        distance_val = 1 - distance_threshold

        # If the maze type is a circle
        if isinstance(maze_polygon, int):

            # Make a point at origin
            p = Point(0, 0)

            # Make an inner circle that is shrunk by distance_val
            inner_circle = maze_polygon * distance_val

            # Buffer by maze_polygon
            polygon_maze = p.buffer(maze_polygon)

            # Make Donut
            reduced_poly = polygon_maze.difference(Point(0.0, 0.0).buffer(inner_circle))

        # For all other maze types, shrink maze down by multiplying by distance_val
        else:

            # make 2 by 2 identity matrix
            ident_mat = np.zeros((2, 2), float)
            np.fill_diagonal(ident_mat, distance_val)
            coords_t = maze_polygon.transpose()
            coords_red = np.matmul(ident_mat, coords_t)
            polygon_maze = Polygon(maze_polygon)
            red_lin = LinearRing(coords_red.transpose())
            reduced_poly = Polygon(maze_polygon, [red_lin])

        return polygon_maze, reduced_poly

    def within_bounds(df, polygon):
        """
        within_bounds(): checks that all coordinates are in Polygon
        Input:
            df (Pandas Dataframe):                  Behavior dataframe with timestamp coordinate data
            polygon (Plotly Polygon):               Polygon to check coordinates against
        """

        # Make column in dataframe and check if the points are in the polygon provided. 0 if in polygon, 1 otherwise
        df['in_poly'] = df.apply(lambda row: polygon.contains(Point(row["Position.X"], row["Position.Y"])), axis=1)

        # All points should be 0. If not all 0, warn user
        if (~df['in_poly']).sum() != 0:
            raise Exception('Warning! Mouse movement detected outside of Polygon boundaries.')

    # Make copy of behavior_df and grab the shape of the maze type provided by user
    df_test = behavior_df.copy()
    shape = shapes(maze_type)

    # Gets inner shape to make distance threshold boundary polygons
    polygon_maze, reduced_poly = shape_shrink(shape, distance_threshold)

    # Checks each row and returns the distance from the edge and True or False if within threshold distance to the wall
    df_test["dist_wall_units"] = df_test.apply(lambda row: polygon_maze.exterior.distance(Point(row["Position.X"], row["Position.Y"])), axis=1)

    # Check each row and returns 1 if within threshold and 0 if not
    df_test['distance_marked'] = df_test.apply(lambda row: 1 if reduced_poly.contains(Point(row["Position.X"], row["Position.Y"])) else 0, axis=1)

    # Checks that all coordinates are in Polygon
    within_bounds(df_test, polygon_maze)

    # Display time taken to run function
    print("FUNC mouse_edge_distance took", str(time.time() - start))

    # Outputs two columns: dist_wall_units and distance_marked
    return df_test['dist_wall_units'], df_test['distance_marked']


def y_maze_regions(behavior_df, maze_type):
    """
    y_maze_regions(): Creates a dictionary of time spent in each region and [[time_diff, region]] master sheet columns
    Input:
        behavior_df (Pandas Dataframe):         The mouse behavioral dataframe
        maze_type (String):                     The type of maze. Must be "Y_Maze" to run analysis. Otherwise, returns 0
    Output:
        mouse_df:           contains columns of time_diff and the region the subject is in
        region_times:       (dictionary) time spent in each y_maze region
    """

    start = time.time()

    mouse_df = behavior_df.copy()
    mouse_df['time_diff'] = mouse_df['#Snapshot Timestamp'].diff()

    # Create region_times dictionary
    region_times = {'y_top': 0.0, 'y_center': 0.0, 'y_left': 0.0, 'y_right': 0.0}

    # Define the polygon regions
    y_top = Polygon(LineString([(-125, 513.924), (-125, 13.925), (125, 13.925), (125, 513.924), (-125, 513.924)]))
    y_center = Polygon(LineString([(0, -202.582), (-125, 13.925), (125, 13.925), (0, -202.582)]))
    y_left = Polygon(LineString([(-125, 13.925), (-558.013, -236.075), (-433.013, -452.582), (0, -202.582), (-125, 13.925)]))
    y_right = Polygon(LineString([(125, 13.925), (558.013, -236.075), (433.013, -452.582), (0, -202.582), (125, 13.925)]))

    # Determines and returns the region a point is in
    def isin_region(row):

        # Return statements for correct region
        if y_top.contains(Point(row["Position.X"], row["Position.Y"])):

            # Ensure that time_diff is not NaN before adding it to region_times
            if not pd.isna(row['time_diff']):
                region_times['y_top'] += row['time_diff']

            return 'y_top'

        elif y_center.contains(Point(row["Position.X"], row["Position.Y"])):

            # Ensure that time_diff is not NaN before adding it to region_times
            if not pd.isna(row['time_diff']):
                region_times['y_center'] += row['time_diff']

            return 'y_center'

        elif y_left.contains(Point(row["Position.X"], row["Position.Y"])):

            # Ensure that time_diff is not NaN before adding it to region_times
            if not pd.isna(row['time_diff']):
                region_times['y_left'] += row['time_diff']

            return 'y_left'

        elif y_right.contains(Point(row["Position.X"], row["Position.Y"])):

            # Ensure that time_diff is not NaN before adding it to region_times
            if not pd.isna(row['time_diff']):
                region_times['y_right'] += row['time_diff']

            return 'y_right'

        else:
            raise Exception(f'Points on {row["#Snapshot Timestamp"]} not found in any region for Y_Maze.')

    if maze_type.lower() == 'y_maze':

        # Determine region by applying isin_region function to each row
        mouse_df['y_region'] = mouse_df.apply(isin_region, axis=1)

        # Iterate through region_times dictionary and round to 3 digits
        for region, time_val in region_times.items():
            region_times[region] = round(time_val, 3)

    else:
        mouse_df['y_region'] = None
        region_times = {'y_top': None, 'y_center': None, 'y_left': None, 'y_right': None}

    print("FUNC y_maze_regions took", str(time.time() - start))


    #return mouse_df[['time_diff', 'region']], region_times
    return mouse_df['y_region'], region_times


def total_distance(behavior_df):
    """
    total_distance():       Calculates and returns total distance travelled during simulation
    Input:
        behavior_df:        Dataframe of behavioral file
    Output:
        distance:           (float) Total distance traveled during simulation
    """

    start = time.time()

    mouse_df = behavior_df.copy()

    # Calculate the difference in X and Y positions
    dx = (mouse_df['Position.X'] - mouse_df['Position.X'].shift())
    dy = (mouse_df['Position.Y'] - mouse_df['Position.Y'].shift())

    # Calculate euclidean distance with a-squared + b-squared = c-squared
    mouse_df['euclidean_dist'] = np.sqrt(dx ** 2 + dy ** 2)

    # Find the sum of euclidean distance
    distance = mouse_df['euclidean_dist'].sum()

    print("FUNC total_distance took", str(time.time() - start))

    return distance


def time_in_custom_regions(behavior_df):
    """
    time_in_custom_regions(): Calculates and returns total amount of time spent in custom regions throughout experiment
    Input:
        behavior_df: Behavior dataframe
    Output:
        custom_region_times_dict: A dictionary keys as custom region keys and values as time spent in each region
    """

    start = time.time()


    # Make temporary copy of dataframe
    df = behavior_df.copy()

    # Create a column of time difference between each timestamp
    df['time_diff'] = df["#Snapshot Timestamp"].diff()

    # Create empty dictionary to hold time spent in each custom region
    region_times_dict = {}

    # Create small, temporary dataframe that contains custom regions and number of timestamps in each
    region_df = pd.DataFrame(df["Trigger Region Identifier"].value_counts())

    # Populate dictionary with custom regions and temporarily set time to zero
    for row_name, row in region_df.iterrows():
        region_times_dict[row_name] = 0.0

    # Iterate through entire experiment df and add time differences to each custom region
    for row_name, row in df.iterrows():
        if row["Trigger Region Identifier"] in region_times_dict:
            region_times_dict[row["Trigger Region Identifier"]] += row["time_diff"]

    # Create empty custom regions dictionary
    custom_region_times_dict = {}

    # Add the word "Custom" to dictionary keys, to differentiate from built-in regions for y_maze
    for key, value in region_times_dict.items():
        #name = "Custom " + str(key)
        name = str(key)
        custom_region_times_dict[name] = round(region_times_dict[key], 3)

    print("FUNC time_in_custom_regions took", str(time.time() - start))


    # Return custom_region_times_dict, even if empty
    return custom_region_times_dict


def get_total_time(behavior_df):
    """
    get_total_time(): Finds total time of experiment in seconds
    Input:
        behavior_df: Behavior dataframe
    Output:
        total_time: total time of experiment in seconds
    """

    start = time.time()


    # Copy the dataframe
    df = behavior_df.copy()

    # Gets final timestamp
    total_time = df["#Snapshot Timestamp"].iloc[-1]

    print("FUNC get_total_time took", str(time.time() - start))

    # Returns final timestamp
    return total_time


# This may be problematic if they change the shape of the maze, so figure that out.
def time_in_center(behavior_df, maze_type):
    """
    time_in_center(): Calculates amount of time spent in center of maze as an indication of anxiety.
    Input:
        behavior_df: behavior dataframe
        maze_type: the type of maze (must be 'circle', 'corridor', 'square', 'y_maze')
    Output:
        time_center_anxiety: Amount of time spent in center of maze (measured in seconds)
        percent_time_center_anxiety: Percent of experiment spent in center of maze (number between 0-1)
    """

    start = time.time()

    # Ensure that maze_type is valid. If not valid, exit function
    possible_mazes = ["circle", "corridor", "square", "y_maze"]
    if maze_type.lower() not in possible_mazes:
        return print("Enter valid maze type: 'circle', 'corridor', 'square', 'y_maze'")

    # Copy behavior_df and create column of time difference between each timestamp
    df = behavior_df.copy()
    df['time_diff'] = df["#Snapshot Timestamp"].diff()

    # By default, the center is defined as the inner third of the maze, except for y_maze
    center_factor = 2/3

    # Set initial time in center to 0
    time_center_anxiety = 0

    # Make df['center_anxiety'] to say either 'center_anxiety' or 'periphery'
    if maze_type.lower() == 'y_maze':

        # Define the points that connect to form the y maze anxiety center
        # Find how to calculate these points automatically
        y_center_anxiety = Polygon(LineString([
            (-41.666, 430.59),
            (41.666, 430.59),
            (41.666, -34.187),
            (444.187, -266.562),
            (402.521, -338.73),
            (0, -106.355),
            (-402.521, -338.73),
            (-444.187, -266.562),
            (-41.666, -34.187),
            (-41.666, 430.59)
        ]))

        def is_in_y_center_anxiety(row):
            """
            is_in_y_center_anxiety(): creates the anxiety center of y maze and determines if within center at timestamp
            Input:
                row: row of behavior dataframe
            Output:
                returns string 'center_anxiety' if in center, 'periphery' if not in center
            """

            # If the x and y coordinates are in the anxiety, return 'center_anxiety'. Otherwise, return 'periphery'
            if y_center_anxiety.contains(Point(row["Position.X"], row["Position.Y"])):
                return 'center_anxiety'
            else:
                return 'periphery'

        # Determine region by applying is_in_y_center_anxiety function to each row
        df['center_anxiety'] = df.apply(is_in_y_center_anxiety, axis=1)

    else:
        # Get distance marked
        df['distance_marked'] = mouse_edge_distance(df, maze_type, center_factor)[1]

        # Make list of locations and set df['center_anxiety'] equal to it
        location = []

        for row_name, row in df.iterrows():
            if row['distance_marked'] == 0:
                location.append('center_anxiety')

            elif row['distance_marked'] == 1:
                location.append('periphery')

        df['center_anxiety'] = location

    # Iterate through rows, adding up time difference if in center
    for row_name, row in df.iloc[1:].iterrows():
        if row['center_anxiety'] == 'center_anxiety':
            time_center_anxiety += row['time_diff']

    # Divide time in center by total time in experiment
    percent_time_center_anxiety = time_center_anxiety / get_total_time(df)

    print("FUNC time_in_center took", str(time.time() - start))

    # Returned rounded times and the df column
    return round(time_center_anxiety, 3), round(percent_time_center_anxiety, 3), df['center_anxiety']



# ---------------------------------------- Master sheet generator ------------------------------------------------ #

# Why only has ymaze and corridor? Needs to work with circle and square mazes
# Needs to figure out what is happening with mouse_edge_distance




def mouse_farm(behavior_path, dist_threshold=0.1):
    """
    mouse_farm(): Organizes the functions to create a dataframe with all the significant variables as columns.
    Input:
        behavior_path (String):                     The filepath of the behavioral file of interest
        dist_threshold (Integer or Float):          Threshold for closeness to wall. By default set to 0.1
    Output:
        mouse_df (Pandas Dataframe):                Dataframe with the comprehensive data from the statistical functions
        metadata_json (Dictionary):                 Dictionary of single line stats summarizing the experiment
    """

    # Time the function
    start_time = time.time()

    # Checks the path for drugs applied - look into this later because there may be more drugs
    def drug_applied(path):
        if 'saline' in path:
            return 'saline'
        elif 'cocaine' in path:
            return 'cocaine'
        else:
            return None

    # makes dataframe from behavior data file
    mouse_df = pd.read_csv(behavior_path, header=2, sep='\t')

    # Reads and makes variables for first two lines of metadata
    settings = pd.read_csv(behavior_path, nrows=2, header=None, sep='\t')
    vrsettings = settings.iloc[0][0]
    mazesettings = settings.iloc[1][0]

    # Finds the maze_array based on the mazesettings in the second line
    maze_types = ['Corridor', 'Y_Maze', 'Square', 'Circle']

    # Ensure that the mazesettings line correctly indicates maze type
    if 'Corridor' not in mazesettings and 'Y_Maze' not in mazesettings and 'Square' not in mazesettings and 'Circle' not in mazesettings:
        return print("Properly indicate maze type immediately before '.maze' of file")

    # Display how much time it took to load the initial part of the function
    print(f'Initial Load: {time.time() - start_time}')

    # Getting desired data for mouse_df, and seeing how long it takes
    filename = os.path.basename(behavior_path)
    maze_array = [maze for maze in maze_types if maze in mazesettings][0]
    drug = drug_applied(behavior_path)
    total_time = get_total_time(mouse_df)
    total_distance_traveled = total_distance(mouse_df[['Position.X', 'Position.Y']])
    speed_instantaneous = calc_speed(mouse_df)
    avg_speed = total_distance_traveled / total_time

    # Load anxiety data with time_in_center() function
    center_anxiety_data = time_in_center(mouse_df, maze_array)
    time_center_anxiety = center_anxiety_data[0]
    percent_time_center_anxiety = center_anxiety_data[1]
    center_anxiety = center_anxiety_data[2]

    # Get information relating to distance to wall
    wall_distance = mouse_edge_distance(mouse_df, maze_array, dist_threshold)

    # Load data relating to custom regions and y regions
    y_maze_data = y_maze_regions(mouse_df, maze_array)

    # Getting dictionary data for metadata_json
    y_region_times = y_maze_data[1]
    custom_regions_dict = time_in_custom_regions(mouse_df) # <- This is a dictionary

    # Add acquired data to mouse_df dataframe
    mouse_df['file_name'] = filename
    mouse_df['file_path'] = behavior_path
    mouse_df['maze_type'] = maze_array
    mouse_df['.vrsettings'] = vrsettings
    mouse_df['.mazesettings'] = mazesettings
    mouse_df['drug_applied'] = drug
    mouse_df['time_diff'] = mouse_df["#Snapshot Timestamp"].diff()
    mouse_df['total_time'] = total_time
    mouse_df['total_distance'] = total_distance_traveled
    mouse_df['speed'] = speed_instantaneous
    mouse_df['avg_speed'] = avg_speed
    mouse_df['y_region'] = y_maze_data[0]
    mouse_df['y_top_time'] = y_region_times['y_top']
    mouse_df['y_center_time'] = y_region_times['y_center']
    mouse_df['y_left_time'] = y_region_times['y_left']
    mouse_df['y_right_time'] = y_region_times['y_right']
    mouse_df['dist_wall_units'] = wall_distance[0]
    mouse_df['center_anxiety'] = center_anxiety
    mouse_df['time_center_anxiety'] = time_center_anxiety
    mouse_df['percent_center'] = percent_time_center_anxiety

    # JSON file of single-line metadata
    metadata_json = {'file_name': filename,
                     'file_path': behavior_path,
                     'maze_type': maze_array,
                     '.vrsetting': vrsettings,
                     '.maze': mazesettings,
                     'drug_applied': drug,
                     'total_time': total_time,
                     'distance_traveled': round(total_distance_traveled, 3),
                     'avg_speed': round(avg_speed, 3),
                     'time_center_anxiety': round(time_center_anxiety, 3),
                     'percent_time_center_anxiety': round(percent_time_center_anxiety, 3),
                     'avg_dist_wall': round(np.mean(wall_distance[0]), 3)
    }

    # If it's a y maze, add y_region_times to metadata_json
    #if maze_array.lower() == "y_maze":
    #    print("It is a y maze")
    metadata_json = {**metadata_json, **y_region_times}

    # If it has custom regions, add the custom regions to metadata_json
    #if bool(custom_regions_dict):
    #print("Has custom regions")
    metadata_json = {**metadata_json, **custom_regions_dict}

    # Display time taken to run function
    print(f'End: {time.time() - start_time}')

    # return mouse_df, metadata_json
    return mouse_df, metadata_json





