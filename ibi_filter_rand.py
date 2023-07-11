import pandas as pd
import numpy as np

def ibi_filter_rand(ibi_data: pd.DataFrame) -> tuple:
    """ IBI filter based on the article by Rand et al (2007): https://pubmed.ncbi.nlm.nih.gov/17518294/
    This filter applies a sliding buffer on the IBI data. 

    -------------------------------------------------------
    Examples:
    1) example of normal behavior
    ibi_values = np.random.normal(.9, .02, 200)  # mean IBI (in seconds), SD IBI, N values
    ibi_df = pd.DataFrame({'Value': ibi_values}, index=[pd.to_datetime('2023-01-01 00:00:00') + pd.Timedelta(seconds=np.round(x, 3)) for x in np.cumsum(ibi_values)]) 
    
    plt.plot(ibi_df)
    plt.plot(ibi_filter_rand(ibi_df)[0], alpha=.5)


    2) example of missed beat
    ibi_values = np.random.normal(.9, .02, 200)  # mean IBI (in seconds), SD IBI, N values
    ibi_df = pd.DataFrame({'Value': ibi_values}, index=[pd.to_datetime('2023-01-01 00:00:00') + pd.Timedelta(seconds=np.round(x, 3)) for x in np.cumsum(ibi_values)]) 
    
    ibi_df2 = ibi_df.drop(ibi_df.iloc[100].name)
    ibi_df2.iloc[100] += ibi_df.iloc[100].item()
    
    plt.plot(ibi_df2)
    plt.plot(ibi_filter_rand(ibi_df2)[0], alpha=.5)


    3) example of split beat
    ibi_values = np.random.normal(.9, .02, 200)  # mean IBI (in seconds), SD IBI, N values
    ibi_df = pd.DataFrame({'Value': ibi_values}, index=[pd.to_datetime('2023-01-01 00:00:00') + pd.Timedelta(seconds=np.round(x, 3)) for x in np.cumsum(ibi_values)]) 
    
    # split a value in half and add a new row to trigger a combine correction
    split_val = ibi_df.iloc[100].item() / 2
    ibi_df.iloc[100] = split_val
    ibi_df.loc[ibi_df.iloc[100].name + pd.Timedelta(seconds=split_val)] = split_val
    ibi_df.sort_index(inplace=True)
    
    plt.plot(ibi_df)
    plt.plot(ibi_filter_rand(ibi_df)[0], alpha=.5)

    -----------------------------------------------------
    :param ibi_data: dataset of IBI data with a pandas datetime index, the IBI data should be in a column named 'Value'
    """
    if len(ibi_data) <= 25:
        raise ValueError(
            'Input dataframe too small, at least 26 rows are required')

    # initialize arrays and dataframes
    ibi_buffer = np.zeros(9, dtype=float)
    averaging_buffer = np.zeros(5, dtype=float)
    ibi_corrected = []
    ibi_list = []
    ibi_copy = ibi_data.copy()

    # set fixed values
    upper_limit = 1.5  # upper limit of IBI
    lower_limit = 0.36  # lower limit of IBI
    correction_count = 0
    counter = 3
    increment = 0
    n_corrected = 0
    n_checked = 0

    for i in range(len(ibi_data) - 5):
        # check if the increment was set, if so skip as many rows as increment states
        if increment > 0:
            increment -= 1
            continue

        # first 20 values are used to find the average IBI
        if i < 20:
            if ibi_copy.Value[i] < upper_limit:
                ibi_list.append(ibi_copy.Value[i])
            start_flag = 1
            continue

        # values 20 to 25 are used to fill the buffers with some specific settings
        if i < 25:
            if start_flag == 1:
                start_flag = 0
                ibi_average = np.mean(ibi_list)
                ibi_buffer[5] = ibi_average
            if np.abs(ibi_buffer[6] - ibi_buffer[5]) <= 0.150:
                correction_count = update_correction_count(correction_count)
                averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
                ibi_buffer = update_ibi_buffer(ibi_buffer, ibi_copy, i, counter)
                ibi_corrected = update_ibi_corrected(ibi_copy, i, ibi_buffer, ibi_corrected)
            else:
                ibi_buffer, averaging_buffer, ibi_corrected, increment, correction_count = correct(ibi_buffer,
                                                                                                   averaging_buffer,
                                                                                                   ibi_copy,
                                                                                                   ibi_corrected,
                                                                                                   correction_count,
                                                                                                   i, counter, 3, 0)
            continue

        # all other values are filtered normally
        if i < len(ibi_copy) - 5:
            n_checked += 1
            threshold_value = compute_threshold(ibi_buffer, 1)

            if np.abs(ibi_buffer[6] - ibi_buffer[5]) < threshold_value:
                correction_count = update_correction_count(correction_count)
                averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
                ibi_buffer = update_ibi_buffer(ibi_buffer, ibi_copy, i, counter)
                ibi_corrected = update_ibi_corrected(ibi_copy, i, ibi_buffer, ibi_corrected)
            else:
                ibi_buffer, averaging_buffer, ibi_corrected, increment, correction_count = correct(ibi_buffer,
                                                                                                   averaging_buffer,
                                                                                                   ibi_copy,
                                                                                                   ibi_corrected,
                                                                                                   correction_count,
                                                                                                   i, counter, 2, 1)
                n_corrected += 1

    ibi_corrected = pd.DataFrame(ibi_corrected, columns=[
                                 'time', 'ibi_corrected', 'correction'])
    ibi_filtered = make_output_df(ibi_corrected, ibi_copy)

    if n_checked == 0:
        q_factor = 0
    else:
        q_factor = (n_checked - n_corrected) / n_checked

    return ibi_filtered, q_factor, ibi_corrected


def decide_correction(ibi_buffer: np.array, ibi_value: float, correct_flag: int, threshold_correct: float) -> int:
    """Decides which of the corrections to apply. The IBI value under consideration is checked against one error type at a time. 
    If the criteria for that error type apply the related correction type is returned
    :param ibi_buffer: list of 9 IBI values
    :param ibi_value: value under consideration
    :param correct_flag: indicates if the value has been corrected by ibi_in_range
    :param threshold_correct: the threshold value to check against
    """
    if correct_flag == 1 and np.abs(ibi_buffer[5] - ibi_value) <= threshold_correct:
        return 1
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6]], 2))[0]) <= threshold_correct:
        return 2
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6]], 3))[0]) <= threshold_correct:
        return 3
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6], ibi_buffer[7]], 1))[0]) <= threshold_correct:
        return 4
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6], ibi_buffer[7]], 2))[0]) <= threshold_correct:
        return 5
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6], ibi_buffer[7]], 3))[0]) <= threshold_correct:
        return 6
    elif np.abs(ibi_buffer[5] - ibi_in_range(correct_value([ibi_buffer[6],
                                                            ibi_buffer[7], ibi_buffer[8]], 3))[0]) <= threshold_correct:
        return 7
    else:
        return 8


def correct(ibi_buffer: np.array, averaging_buffer: np.array, ibi_data: pd.DataFrame, ibi_corrected: list, correction_count: int, index: int,
            counter: int, flag_threshold: int, average_mode: int) -> tuple:
    """
    Corrects the IBI signal based on a selection of possible mistakes. 
    If more than 3 corrections have already been applied the function replaces the new value with the mean and subtracts 1 from the correction count
    :param ibi_buffer: list of 9 IBI values
    :param averaging_buffer: list of 5 IBI values
    :param ibi_data: Dataframe of IBI values
    :param ibi_corrected: list of checked IBI values
    :param correction_count: number of consecutive corrections applied
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param flag_threshold: indicator of which threshold method to use
    :param average_mode: indicator of which average method to use
    """

    if not isinstance(ibi_data.index, pd.DatetimeIndex):
        raise TypeError(
            'IBI data needs to have an datetime index, use pd.DatetimeIndex on the index to fix this issue')

    # increment can add to i in the while loop, this enables skipping a row
    increment = 0

    # correct applies a correction as long as there are no more than 3 corrections necessary in a row
    if correction_count < 3:
        # first find the threshold used to test against
        threshold_correct = compute_threshold(ibi_buffer, flag_threshold)

        # extract the value we want to check and if ibi_in_range had to correct the value
        ibi_value, correct_flag = ibi_in_range(ibi_buffer[6])
        correction_type = decide_correction(ibi_buffer, ibi_value, correct_flag, threshold_correct)

        # first case: value is corrected and is now below the threshold
        if correction_type == 1:
            averaging_buffer = update_averaging_buffer(
                averaging_buffer, ibi_buffer, 1)
            ibi_buffer, ibi_corrected = correction_1(ibi_buffer, ibi_data, index, counter, ibi_corrected, ibi_value)
            increment = 0

        # second case: missed heart beat
        elif correction_type == 2:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
            ibi_buffer, ibi_corrected = correction_2(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 0

        # third case: two missed heart beats
        elif correction_type == 3:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
            ibi_buffer, ibi_corrected = correction_3(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 0

        # fourth case: false trigger, add two heart beats together
        elif correction_type == 4:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 2)
            ibi_buffer, ibi_corrected = correction_4(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 1

        # fifth case: false trigger replace 2 with average of 2 beats
        elif correction_type == 5:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 2)
            ibi_buffer, ibi_corrected = correction_5(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 1

        # sixth case: false trigger replace 3 with average of 2 beats
        elif correction_type == 6:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 2)
            ibi_buffer, ibi_corrected = correction_6(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 1

        # seventh fix: false trigger replace 3 with average of 3 beats
        elif correction_type == 7:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 3)
            ibi_buffer, ibi_corrected = correction_7(ibi_buffer, ibi_data, index, counter, ibi_corrected)
            increment = 2

        # Eighth case: IBI appears to be faulty but no correction applies, correct with average
        else:
            averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
            ibi_buffer, ibi_corrected = correction_8(ibi_buffer, averaging_buffer, ibi_data, index,
                                                     counter, ibi_corrected, average_mode)
            increment = 0

        # add 1 to the correction count
        correction_count += 1

    # correction count is at 3, correct with the average and remove 1 from correction count
    else:
        averaging_buffer = update_averaging_buffer(averaging_buffer, ibi_buffer, 1)
        correction_count = update_correction_count(correction_count)
        ibi_buffer, ibi_corrected = correction_9(ibi_buffer, averaging_buffer, ibi_data, index,
                                                 counter, ibi_corrected, average_mode)
        increment = 0

    return ibi_buffer, averaging_buffer, ibi_corrected, increment, correction_count

    
    
# corrections
def correction_1(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list, ibi_value: float) -> tuple:
    """ Basic threshold correction was enough to correct the value
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    :param ibi_value: value under consideration
    """
    # shift the ibi_buffer one place to the left and fill in the (corrected) value
    ibi_buffer = np.roll(ibi_buffer, -1)
    ibi_buffer[5] = ibi_value
    ibi_buffer[8] = ibi_data.Value[index+counter]

    # fill corrected IBI dataframe
    new_row = [ibi_data.index[index], ibi_buffer[5], 1]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_2(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ Correction for a single missed heart beat 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # get the correction value and update the ibi buffer
    store_value = ibi_in_range(correct_value([ibi_buffer[6]], 2))[0]
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5:7] = [store_value, store_value]

    # update the time
    time_difference = (ibi_data.index[index] - ibi_data.index[index-1]) / 2

    # update first corrected IBI dataframe
    new_row = [ibi_data.index[index - 1] + time_difference, ibi_buffer[5], 2]
    ibi_corrected.append(new_row)

    # update second correction
    ibi_buffer = np.roll(ibi_buffer, -1)
    ibi_buffer[8] = ibi_data.Value[index+counter]
    new_row = [ibi_data.index[index], ibi_buffer[5], 2]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_3(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ Correction for two missed heart beats 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # get the correction value and update the ibi buffer
    store_value = ibi_in_range(correct_value([ibi_buffer[6]], 3))[0]
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -2)
    ibi_buffer[4:7] = [store_value, store_value, store_value]

    # update the time
    time_difference = (ibi_data.index[index] - ibi_data.index[index-1]) / 3

    # update first corrected IBI dataframe
    new_row = [ibi_data.index[index - 1] + time_difference, ibi_buffer[5], 2]
    ibi_corrected.append(new_row)

    # update second corrected IBI dataframe
    new_row = [ibi_data.index[index - 1] +
               time_difference * 2, ibi_buffer[5], 2]
    ibi_corrected.append(new_row)

    # update third correction
    ibi_buffer = np.roll(ibi_buffer, -1)
    ibi_buffer[8] = ibi_data.Value[index+counter]
    new_row = [ibi_data.index[index], ibi_buffer[5], 2]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_4(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ False trigger, add two heart beats together 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # update ibi_buffer
    ibi_buffer[6] = ibi_in_range(correct_value(
        [ibi_buffer[6], ibi_buffer[7]], 1))[0]
    ibi_buffer[0:7] = np.roll(ibi_buffer[0:7], -1)
    ibi_buffer[6] = ibi_buffer[8]
    # replace last 2 with the new beat
    ibi_buffer[7] = ibi_data.Value[index + counter]
    ibi_buffer[8] = ibi_data.Value[index + counter + 1]

    # update correction df
    new_row = [ibi_data.index[index + 1], ibi_buffer[5], 4]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_5(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ False trigger replace 2 with average of 2 beats 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # update ibi_buffer
    store_value = ibi_in_range(correct_value(
        [ibi_buffer[6], ibi_buffer[7]], 2))[0]
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value

    # update correction df
    new_row = [ibi_data.index[index], ibi_buffer[5], 5]
    ibi_corrected.append(new_row)

    # prepare next update
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value
    ibi_buffer[6] = ibi_buffer[8]
    ibi_buffer[7] = ibi_data.Value[index + counter]
    ibi_buffer[8] = ibi_data.Value[index + counter + 1]

    # update correction df
    new_row = [ibi_data.index[index + 1], ibi_buffer[5], 5]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_6(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ False trigger replace 3 with average of 2 beats 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # update ibi_buffer
    store_value = ibi_in_range(correct_value(
        [ibi_buffer[6], ibi_buffer[7]], 3))[0]
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value

    # update the time
    time_difference = (ibi_data.index[index+1] - ibi_data.index[index-1]) / 3

    # update correction df
    new_row = [ibi_data.index[index - 1] + time_difference, ibi_buffer[5], 6]
    ibi_corrected.append(new_row)

    # second update
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value
    new_row = [ibi_data.index[index - 1] +
               time_difference * 2, ibi_buffer[5], 6]
    ibi_corrected.append(new_row)

    # third update
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value
    ibi_buffer[6] = ibi_buffer[8]
    ibi_buffer[7] = ibi_data.Value[index + counter]
    ibi_buffer[8] = ibi_data.Value[index + (counter+1)]
    new_row = [ibi_data.index[index + 1], ibi_buffer[5], 6]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_7(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list) -> tuple:
    """ False trigger replace 3 with average of 3 beats 
    :param ibi_buffer: list of 9 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    """
    # update ibi_buffer
    store_value = ibi_in_range(correct_value(
        [ibi_buffer[6], ibi_buffer[7], ibi_buffer[8]], 3))[0]
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value

    # update correction df
    new_row = [ibi_data.index[index], ibi_buffer[5], 7]
    ibi_corrected.append(new_row)

    # second update
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value
    new_row = [ibi_data.index[index + 1], ibi_buffer[5], 7]
    ibi_corrected.append(new_row)

    # third update
    ibi_buffer[0:6] = np.roll(ibi_buffer[0:6], -1)
    ibi_buffer[5] = store_value
    ibi_buffer[6] = ibi_data.Value[index + counter]
    ibi_buffer[7] = ibi_data.Value[index + (counter+1)]
    ibi_buffer[8] = ibi_data.Value[index + (counter+2)]
    new_row = [ibi_data.index[index + 2], ibi_buffer[5], 7]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_8(ibi_buffer: np.array, averaging_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list, average_mode: int) -> tuple:
    """ Unknown issue, correct with average 
    :param ibi_buffer: list of 9 IBI values
    :param averaging_buffer: list of 5 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    :param average_mode: 0 if the new value should be the previous value, 1 if the new value should be the mean of the average buffer
    """
    if average_mode == 0:
        ibi_buffer[6] = ibi_buffer[5]
        ibi_buffer = np.roll(ibi_buffer, -1)

    elif average_mode == 1:
        store_average = np.array(averaging_buffer).mean()
        ibi_buffer[6] = store_average
        ibi_buffer = np.roll(ibi_buffer, -1)

    # update correction df
    ibi_buffer[8] = ibi_data.Value[index + counter]
    new_row = [ibi_data.index[index], ibi_buffer[5], 8]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


def correction_9(ibi_buffer: np.array, averaging_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int, ibi_corrected: list, average_mode: int) -> tuple:
    """ Unknown issue, correct with average 
    :param ibi_buffer: list of 9 IBI values
    :param averaging_buffer: list of 5 IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param counter: the number of rows to skip
    :param ibi_corrected: list of checked IBI values
    :param average_mode: 0 if the new value should be the previous value, 1 if the new value should be the mean of the average buffer
    """
    if average_mode == 0:  # average is just the previous value
        ibi_buffer[6] = ibi_buffer[5]
        ibi_buffer = np.roll(ibi_buffer, -1)
    else:  # average is the mean of the buffer array
        store_average = np.array(averaging_buffer).mean()
        ibi_buffer[6] = store_average
        ibi_buffer = np.roll(ibi_buffer, -1)

    # update correction df
    ibi_buffer[8] = ibi_data.Value[index + counter]
    new_row = [ibi_data.index[index], ibi_buffer[5], 9]
    ibi_corrected.append(new_row)

    return ibi_buffer, ibi_corrected


# threshold functions
def compute_threshold(ibi_buffer: np.array, flag_threshold: int) -> float:
    """ Returns the threshold value. 
    The function supports 3 different methods, indicated by the flag_threshold argument 
    :param ibi_buffer: array of IBI values to calculate the threshold on
    :param flag_threshold: setting for which method to use [1 | 2 | 3]
    """
    if flag_threshold not in [1, 2, 3]:
        raise ValueError('Flag threshold needs to be either 1, 2 or 3')
    if flag_threshold == 1:
        threshold_val = compute_threshval(ibi_buffer, 10)
        if threshold_val < 0.05:
            threshold_val = 0.05
        elif threshold_val > 0.20:
            threshold_val = 0.2
    elif flag_threshold == 2:
        threshold_val = compute_threshval(ibi_buffer, 25)
        threshold_val = np.where(threshold_val < 0.01, 0.01, 0.1).item()
    elif flag_threshold == 3:
        threshold_val = 0.1
    return threshold_val


def compute_threshval(ibi_buffer: np.array, multiplier: int) -> float:
    """ Computes the specific threshold value within compute_threshold.
    This threshold calculates the mean succesive difference of the trusted values in the buffer
    :param ibi_buffer: array of IBI values to calculate the threshold on
    :param multiplier: value to scale the mean succesive difference
    """
    threshold_array = []
    threshold_array = [np.abs(ibi_buffer[i] - ibi_buffer[i+1]) for i in range(0, 4, 1)]
    threshold_val = (np.sum(threshold_array) * multiplier / 4)
    return threshold_val


# update functions
def update_correction_count(correction_count: int) -> int:
    """ Updates the correction_count by subtracting 1 from the count, unless it is already 0
    :param correction_count: number indicating how many corrections have been applied
    """
    return np.where(correction_count > 0, correction_count - 1, 0)


def update_ibi_buffer(ibi_buffer: np.array, ibi_data: pd.DataFrame, index: int, counter: int) -> np.array:
    """ Rolls the IBI buffer n steps forward
    :param ibi_buffer: array of IBI values
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the buffer
    :param counter: the number of rows to skip
    """
    ibi_buffer[6] = ibi_in_range(ibi_buffer[6])[0]
    ibi_buffer = np.roll(ibi_buffer, -1)
    ibi_buffer[-1] = ibi_data.Value[index + counter]
    return ibi_buffer


def update_ibi_corrected(ibi_data: pd.DataFrame, index: int, ibi_buffer: np.array, ibi_corrected: list) -> list:
    """ Updates the corrected ibi dataframe with the new values 
    :param ibi_data: Dataframe of IBI values
    :param index: the index of the new IBI value to add to the corrected IBI values
    :param ibi_buffer: array of IBI values
    :param ibi_corrected: list of checked IBI values
    """
    new_row = [ibi_data.index[index], ibi_buffer[5], 0]
    ibi_corrected.append(new_row)
    return ibi_corrected


def update_averaging_buffer(averaging_buffer: np.array, ibi_buffer: np.array, n:int) -> np.array:
    """ Updates the averaging buffer by removing the first value and adding in the new value from the ibi buffer
    :param averaging_buffer: list of 5 values, used to get an average if no correction can be applied
    :param ibi_buffer: list of 9 IBI values
    :param n: integer specifying how many values to update
    """
    averaging_buffer = np.roll(averaging_buffer, -n)
    averaging_buffer[-n:] = [ibi_in_range(ibi_buffer[i + 6])[0]
                             for i in range(n)]
    return averaging_buffer
    

# helper functions
def make_output_df(corrected_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """ Reshapes the filtered data into the original shape, including the last values 
    :param corrected_df: the Dataframe of corrected IBI values
    :param original_df: the Dataframe of original IBI values
    """
    last_values = original_df.loc[corrected_df.time.iloc[-1]:].iloc[1:]
    corrected_values = corrected_df.loc[:,['time', 'ibi_corrected']].copy()
    corrected_values.set_index('time', inplace=True)
    corrected_values.columns = ['Value']
    
    return pd.concat([corrected_values, last_values], axis=0).sort_index()


def correct_value(value_list: list, divider: int) -> float:
    """ Performs all possible correction, see paper for description:
    split: single value in value_list / 2
    split_3: single value in value_list / 3
    combine: 2 values in value_list / 1
    combine2_split2: 2 values in value_list / 2
    combine2_split3: 2 values in value_list / 3
    combine3_split3: 3 values in value_list / 3
    :param value_list: the list of IBI values that are used to correct the value
    :param divider: integer that indicates what the sum of the value list should be devided by
    """
    if len(value_list) > 3:
        raise ValueError(
            'Value list cannot contain more than 3 values. See original paper')
    if divider > 3 or divider < 1:
        raise ValueError('Divider must be either 1, 2 or 3')
    return np.sum(value_list) / divider


def ibi_in_range(value: float, lower: float = 0.36, upper: float = 1.5) -> tuple:
    """ Corrects the IBI value if it is lower than lower limit and higher than the upper limit 
    :param value: the IBI value that needs to be checked
    :param lower: the lowest IBI value we accept
    :param upper: the highest IBI value we accept
    """
    if np.logical_and(value >= lower, value <= upper):
        return value, 0
    elif value > upper:
        return upper, 1
    elif value < lower:
        return lower, 1
