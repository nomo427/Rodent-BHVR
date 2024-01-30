import matplotlib.pyplot as plt
import numpy as np
import updated_helper_functions
import seaborn as sns
import pandas as pd
import statistics
import os
import time
from PIL import Image


# Dictionary that stores all the image types that will be saved
img_types = {
    '.svg': 'SVG',
    '.jpg': 'JPG',
    '.png': 'PNG',
}


class MouseExperiment:

    # Construct and initialize attributes
    def __init__(self, behavioral_path):
        self.mouse_data = updated_helper_functions.mouse_farm(behavioral_path)
        self.df = self.mouse_data[0]
        self.metadata = self.mouse_data[1]
        self.shape = self.metadata['maze_type']
        self.filepath = self.metadata['file_path']
        self.metadata_df = pd.DataFrame([self.mouse_data[1]])
        self.title = self.filepath.split("/")[-1]


    def save_as(self, output_dir=None):
        """
        save_as(): Saves the processed dataframe as a csv
        Input:
            output_dir (string):            Absolute filepath where to save the csv
        Output:
            NA
        """

        # If no output_dir is provided, save it in the same location as original behavior file, with "_proceessed.csv"
        if output_dir is None:
            pathname = os.path.splitext(self.filepath)[0]
            self.df.to_csv(f'{pathname}_processed.csv')
            self.metadata_df.to_json(f'{pathname}_additional_data.json', orient="index")
            print("File processed")

        # If there is an output_dir provided, save to that location with "_processed.csv"
        elif output_dir is not None:
            self.df.to_csv(f'{output_dir}_processed.csv')
            self.metadata_df.to_json(f'{output_dir}_experiment_data.json', orient="index")
            print("File processed")


    def list_drugs(self):
        """
        list_drugs(): Returns a list of all drugs administered in experiment
        Input:
            NA
        Output:
            List of all unique drugs applied in the experiment
        """

        # Return list of drugs used in experiment
        return self.df['drug_applied'].unique()


    def overall_time_spent_near_wall(self, units=5):
        """
        overall_time_spent_near_wall: Returns dictionary summarizing closeness to wall.
        Input:
            units (Integer or Float):           Number of units
        Output:
            wall_dict (Dictionary):             A dictionary summarizing closeness to wall
        """

        # Make a copy of self.df
        mouse_df = self.df

        # Make a column 'distance_marked' with 1 if distance to wall is less than desired threshold, and 0 if not
        self.df['distance_marked'] = np.where(mouse_df['dist_wall_units'] < 34 + units, 1, 0)
        sum_ = mouse_df[mouse_df['distance_marked'] == 1]['time_diff'].sum()
        wall_percent = sum_ / mouse_df['#Snapshot Timestamp'].max()

        # Create dictionary
        wall_dict = {'seconds_near_wall': sum_, 'proportion_near_wall': wall_percent, 'units': units}

        # Return dictionary
        return wall_dict


    def movement_plot(self, region=None, save=False, save_filepath=None):
        """Shows the subject's movement. Can customize by setting 'region' to:
         'custom_region', 'center_anxiety', and 'y_region'."""

        start = time.time()

        # Obtain dataframe
        df = self.df

        fig = plt.figure(dpi=125)

        ax1 = fig.add_subplot(111)
        bound_x, bound_y = updated_helper_functions.shapes(self.shape, plotly=True)

        # Set set_hue to appropriate column of df to distinguish
        set_hue = ""
        extension = ""
        if region is None:
            set_hue = ""
            extension = "_movement_plot"
        elif region.lower() == 'custom_region':
            set_hue = 'Trigger Region Identifier'
            extension = '_custom_region'
        elif region.lower() == 'center_anxiety':
            set_hue = 'center_anxiety'
            extension = '_center_anxiety'
        elif region.lower() == 'y_region':
            set_hue = '_y_region'
            extension = 'y_region'
        elif region.lower() == 'speed':
            set_hue = 'speed'
            extension = '_speed'

        # Ensure that argument for region parameter is valid. yes
        regions = ['custom_region', 'center_anxiety', 'y_region', 'speed']
        if region is not None and region.lower() not in regions:
            return print("Please input a valid region type to visualize. Options are: "
                         "'custom_region', 'center_anxiety', and 'y_region'.")

        # If maze type is a y maze
        if self.shape.lower() == 'y_maze':
            # Set correct hue
            if region is None:
                plt.plot(df["Position.X"], df["Position.Y"], linewidth=0.5)
            else:
                fig = sns.scatterplot(x='Position.X', y='Position.Y', hue=set_hue, data=df, size=0.1, legend=False)

            sep_x = [0, -125, 125, 0]
            sep_y = [-202.582, 13.925, 13.925, -202.582]

            ax1.plot(sep_x, sep_y)

        # If maze type is not a y maze
        else:
            # Ensure that region is not set to 'y_region'
            if region is not None and region.lower() == 'y_region':
                return print('Maze is not a Y_Maze. Please enter a valid region to visualize.')

            # Set correct hue
            if region is None:
                plt.plot(df["Position.X"], df["Position.Y"], linewidth=0.5)
            else:
                #fig = sns.scatterplot(x='Position.X', y='Position.Y', hue=set_hue, data=df, size=0.1)
                fig = sns.scatterplot(x='Position.X', y='Position.Y', hue=set_hue, data=df)

        plt.xlabel("Position.X")
        plt.ylabel("Position.Y")

        ax1.plot(bound_x, bound_y)
        figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

        # Add title
        title = self.filepath.split("/")[-1]
        plt.title(title)

        # Save the movement plot if indicated
        if save:

            # Loop through dictionary of img_types and save all the types of images for each plot
            for img_type_key, img_type_value in img_types.items():
                name = save_filepath[:-9] + extension + img_type_key
                print("Trying to save to:", name)

                # Save to Images folder
                plt.savefig(name, format=img_type_value)

                # If it is png, resize it as well
                if img_type_value == "PNG":

                    # Resize the image and save that as well
                    display = Image.open(name)
                    display.thumbnail(size=(500, 500))
                    sized_named = name[:-4] + '_sized' + img_type_key
                    display.save(sized_named, quality=888)

                print("Saved")

            print("movement_plot Auto save: ", time.time() - start)

            plt.clf()


        else:

            print("movement_plot: ", time.time() - start)
            #plt.show()
            return plt.gcf()


    # in tutorial, mention something about how filepath needs '/'
    # Need to get the maze boundaries into the plot
    def heatmap(self, num_bin=20, save=False, save_filepath=None, maze_outline=True):
        """Shows a 2d histplot of where subject spent most time. Can customize by adjusting 'num_bin' parameter"""

        start = time.time()

        # Grab the title of the experiment
        title = self.filepath.split("/")[-1]

        fig = plt.figure(dpi=125)
        ax1 = fig.add_subplot(111)
        bound_x, bound_y = updated_helper_functions.shapes(self.shape, plotly=True)

        print("bound_x", bound_x, "bound_y", bound_y)

        if self.shape.lower() == 'y_maze':
            # Imposes the maze boundaries on the plot
            if maze_outline:

                plt.hist2d(self.df['Position.X'],
                           self.df['Position.Y'],
                           bins=[np.linspace(-600,600,num_bin), np.linspace(-600,600,num_bin)],
                )

                sep_x = [0, -125, 125, 0]
                sep_y = [-202.582, 13.925, 13.925, -202.582]

                ax1.plot(sep_x, sep_y)

                ax1.plot(bound_x, bound_y)
                figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds

            # Does not impose the maze boundaries on the plot
            else:
                plt.hist2d(self.df['Position.X'],
                           self.df['Position.Y'],
                           bins=[np.linspace(-600,600,num_bin), np.linspace(-600,600,num_bin)],
                )

        # If any other type of maze
        elif self.shape.lower() == 'corridor':

            # Make the heatmap
            plt.hist2d(self.df['Position.X'],
                       self.df['Position.Y'],
                       bins=[np.linspace(min(bound_x) - 5, max(bound_x) + 5, num_bin),
                             np.linspace(min(bound_y) - 5, max(bound_y) + 5, num_bin)],
            )

            # If indicated, add the maze outline
            if maze_outline:
                ax1.plot(bound_x, bound_y, color='r')
                figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds




        # APPLIES TO ALL HEATMAPS
        # Add title, colorbar, and x,y labels
        plt.colorbar()
        plt.title(title)
        plt.xlabel("Position.X")
        plt.ylabel("Position.Y")

        # Save the heatmap if indicated. This is only used by user command, not the GUI
        if save:

            # Loop through dictionary of img_types and save all the types of images for each plot
            for img_type_key, img_type_value in img_types.items():
                name = save_filepath[:-9] + "_heatmap" + img_type_key
                print("Trying to save to:", name)

                # Save to Images folder
                plt.savefig(name, format=img_type_value)

                # If it is png, resize it as well
                if img_type_value == "PNG":
                    # Resize the image and save that as well
                    display = Image.open(name)
                    display.thumbnail(size=(500, 500))
                    sized_named = name[:-4] + '_sized' + img_type_key
                    display.save(sized_named, quality=888)

                print("Saved")

            print("movement_plot Auto save: ", time.time() - start)

            plt.clf()
            #return plt.gcf()


        else:
            # Quoted out for testing
            #plt.show()
            print("heatmap: ", time.time() - start)
            #plt.show()
            return plt.gcf()

    def speed_plot(self, save=False, save_filepath=None, median=True, average=True):
        """
        speed_plot(): Graphs the speed of the mouse over the experiment
        input:
            median:     (bool) indicates if median speed should be displayed
            average:    (bool) indicates if average speed should be displayed
        output:
            Plot of the speed of mouse, compared to average for the experiment
        """

        start = time.time()

        # Plot the speed of the experiment over duration of the experiment
        plt.plot(self.df["#Snapshot Timestamp"], self.df["speed"], c="green", linewidth=0.5)

        # Determine and plot the median speed as a horizontal line
        if average:
            avg = sum(self.df["speed"]) / len(self.df["speed"])
            plt.axhline(y=avg, color='b', linestyle='-', linewidth=2, label="Average")

        # Determine and plot the median speed as a horizontal line
        if median:
            med = statistics.median(self.df["speed"])
            plt.axhline(y=med, color='r', linestyle='-', linewidth=2, label="Median")

        # Add title of the experiment
        plt.title(self.title)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Speed (units / second)")
        plt.legend()

        # Save the heatmap if indicated. This is only used by user command, not the GUI
        if save:

            # Loop through dictionary of img_types and save all the types of images for each plot
            for img_type_key, img_type_value in img_types.items():
                name = save_filepath[:-9] + "_speed" + img_type_key
                print("Trying to save to:", name)

                # Save to Images folder
                plt.savefig(name, format=img_type_value)

                # If it is png, resize it as well
                if img_type_value == "PNG":
                    # Resize the image and save that as well
                    display = Image.open(name)
                    display.thumbnail(size=(500, 500))
                    sized_named = name[:-4] + '_sized' + img_type_key
                    display.save(sized_named, quality=888)

                print("Saved")

            print("movement_plot Auto save: ", time.time() - start)

            plt.clf()
            # return plt.gcf()

        else:

            print("speed_plot: ", time.time() - start)
            #plt.show()
            return plt.gcf()


    def wall_distance_plot(self, save=False, save_filepath=None):
        """
        wall_distance_plot(): Graphs the speed of the mouse over the experiment
        input:
            save (Bool):                        Indicates if plot should be saved or not
            save_filepath (String):             Absolute filepath of where plot should be saved
        output:
            Plot
        """

        # Time the function
        start = time.time()

        # Plot the speed of the experiment over duration of the experiment
        plt.plot(self.df['#Snapshot Timestamp'], self.df['dist_wall_units'], c='g', linewidth=0.5)
        plt.xlabel("Time (sec.)")
        plt.ylabel("Distance to Nearest Wall (Unit)")
        plt.title(self.title)

        if save:

            # Loop through dictionary of img_types and save all the types of images for each plot
            for img_type_key, img_type_value in img_types.items():
                name = save_filepath[:-9] + "_wall_distance" + img_type_key
                print("Trying to save to:", name)

                # Save to Images folder
                plt.savefig(name, format=img_type_value)

                # If it is png, resize it as well
                if img_type_value == "PNG":

                    # Resize the image and save that as well
                    display = Image.open(name)
                    display.thumbnail(size=(500, 500))
                    sized_named = name[:-4] + '_sized' + img_type_key
                    display.save(sized_named, quality=888)

                print("Saved")

            # Display the time taken to run function
            print("wall_distance_plot Auto save: ", time.time() - start)

            # Clear the plot to avoid having multiple plots overlapping with each other
            plt.clf()

        # If the plot is not going to be saved, just return the plt figure
        else:

            # Return plt figure
            return plt.gcf()

        # Display the time taken to run function
        print("wall_distance_plot: ", time.time() - start)


    def custom_regions_barplot(self, save=False, save_filepath=None):
        """
        custom_regions_barplot(): Makes a barplot for the times spent in each custom region and y_region
        Input:
            save (Bool):                    Indicates whether it should save plot or not
            save_filepath (String):         Absolute filepath where to save the plot to
        Output:
            Barplot
        """

        # Grab list of regions present in experiment and list of times spent in each
        regions = list(self.metadata_df.columns)[12:]
        times = list(self.metadata_df.iloc[0,12:])

        # Make bar plot of regions and times
        plt.bar(regions, times, label=times)

        # Show the plot
        plt.show()






if __name__ == "__main__":
    rytz = MouseExperiment('/Users/noahmoffat/Documents/Research/Yorgason_Lab/VR/Raw Data/CPP Experiment Data/Final Test Day/8010_CPP_y_maze__1.behavior')
    #rytz.movement_plot()
    #print(rytz.df.head())
    #rytz.movement_plot(region='speed')
    #rytz.heatmap()
    #rytz.wall_distance_plot()
    rytz.custom_regions_barplot()


    # Figure out how to do batch analysis
    # Find the best experiment file to use to try extracting boundaries from

