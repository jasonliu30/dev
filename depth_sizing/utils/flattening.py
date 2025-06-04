from scipy.signal import savgol_filter
import numba
import numpy as np
from ToF import interpolate_indications

def interpolate_indications(lag, indications):
    """""
    
    This removes indications from a frame of lags, the indications should be a list of booleans
    (True where there is an indication)
    The function removes all areas that are True in indications, 
    and does linear interpolation to replace those values
    Inputs
    --------
    -lag: 1D integer array of lags in shape (number of A-scans)
    
    -indications: 1D boolean array of indication locations in shape (number of A-scans)
 
    Outputs
    --------
    -lag: 1D integer array of lags with indications removed in shape (number of A-scans)
    """

    i = 1
    new_lags = [i for i in lag]
    in_indication = False
    start_of_indication = -1
    while i < len(lag):
        if in_indication and not indications[i]:
            new_values = np.interp(range(start_of_indication, i), [start_of_indication, i],
                                   [lag[start_of_indication], lag[i]])
            new_lags[start_of_indication:i] = new_values
            in_indication = False
        if indications[i] and not in_indication:
            start_of_indication = i
            in_indication = True
        i += 1

    return np.array(new_lags)


def Smoothlags(lags: np.ndarray, sd_limit: int, MaxIter: int, SG_ord=int(5), SG_flen=int(301), OUTLIER_GRADIENT = True) -> np.ndarray:
    """
    Function to find the general trend of the lags in argument "corslags".
    This is done by repeatedly passing the lags through a filter to smooth the lags and removing outliers before smoothing again.
    After a certain number of iterations, the general trend of the lags (i.e. the smoothed lags) will emerge.
    Parameters
    ----------
    lags : np.ndarray
        Numpy array with the lags.
    sd_limit : list
        The number of standard deviations the residual (i.e. difference between the original and smoothed lags)
        must be before it is considered an outlier.
    MaxIter : int
        The number of times the lags will be smoothed to determine the general trend.
    SG_ord : int
        Order used for the SG filter fit
    SG_flen : int
        Filter length used for the SG filter
    Returns
    -------
    Smoothed signal
    """
    # Get the maximum number of iterations and the limit
    Limit = sd_limit;
    Vector = lags*-1;
    NotInclude = np.full(Vector.shape[0], False, dtype=np.bool);

    # Loop through the number of times
    for i in range(MaxIter):
        # Obtain the signal without any identified outliers
        # Outliers are removed and linear interpolation is used to fill in the gaps for filtering
        SignalNoOutliers = np.array(interpolate_indications(Vector, NotInclude));

        # Smooth the signal with the selected filter and calculate the residuals
        Smoothed = savgol_filter_padding(SignalNoOutliers, SG_flen, SG_ord)

        Residuals = SignalNoOutliers - Smoothed;

        # Calculate the standard error that will be used to set the limits for the next iteration of smoothing
        Sigma = np.sqrt(np.square(Residuals).sum() / Residuals.shape[0]);

        # Find the outliers
        Outliers = Residuals < (- Limit * Sigma);
        # Now remove the points around the outliers that make up the trough
        Outliers = OutlierExtent(Outliers, Residuals,Gradient=OUTLIER_GRADIENT);

        # Remove the points from the smoothing process
        NotInclude = NotInclude | Outliers;

    # Return the residuals
    return Smoothed*-1,NotInclude


def Smoothlags_remove_flaws(lags: np.ndarray, sd_limit: int, MaxIter: int,indications:np.ndarray, SG_ord=int(5), SG_flen=int(301), OUTLIER_GRADIENT = True) -> np.ndarray:
    
    lags_without_indications = np.array(interpolate_indications(lags,indications))
    Smoothed,not_include = Smoothlags(lags_without_indications, sd_limit, MaxIter, SG_ord, SG_flen, OUTLIER_GRADIENT)
    return Smoothed,not_include | indications


@numba.njit
def CalcGradient(Array1D: np.ndarray):
    
    """
    Calculates the gradient using the 2nd order finite difference method.
    Gradient at edge of array is first order.
    Assumes the array is evenly spaced.
    
    Args :
        Array1D: 1d array
    Returns:
        Result: Return the calculated gradients
    
    """

    # Initialize the array that will hold the gradient
    Result = np.empty(Array1D.shape);

    # Calculate the non-edge gradients
    Result[1:(Result.shape[0] - 1)] = (Array1D[2:] - Array1D[:(Array1D.shape[0] - 2)]) / 2;

    # Calculate the edge gradients
    Result[0] = Array1D[1] - Array1D[0];
    Result[-1] = Array1D[-1] - Array1D[-2];

    # Return the calculated gradients;
    return Result;

@numba.njit
def OutlierExtent(Passed: np.ndarray, Residuals: np.ndarray,  Gradient: bool =False) -> np.ndarray:
    """
    In the function SmoothCor(), the difference between the original correlations and the correlations smoothed by function Filter() is calculated (i.e. the residuals).
    Any location where the original correlations is significantly less than the smoothed correlation (i.e. an outlier) is indicative of an indication, and this is marked by False in the Numpy array Passed.
    This function marks the entire trough that contains the outlier as False as well.
    The extent of the trough is when the residuals become positive (i.e. the original correlation is greater than the smoothed correlation).
    Parameters
    ----------
    Passed : np.ndarray
        Numpy array of booleans where False indicates the location is initial considered an outlier, where it indicates an indication.
        The function modifies array in place to switch elements to False that are considered part of the trough.
    Residuals : np.ndarray
        Numpy array with the values of the residuals between the actual and filtered correlation.
    Returns
    -------
    Indications array of booleans where True is the extent of the indications.
    """
    # Find the indices where outliers are identified
    Outliers = np.where(Passed)[0];

    # Calculate the gradient of the residuals if needed
    if Gradient:
        ResidGrad = CalcGradient(Residuals);
    else:
        ResidGrad = np.zeros(Residuals.shape);

    # Loop through all the identified outliers, and remove all points until the residuals are greater than zero.
    # This marks the extent of the trough that should be removed from the flattening.
    for i in Outliers:
        # Loop behind to find the extent of the trough
        Index = i - 1;
        while Index >= 0 and (not Passed[Index]) and (Residuals[Index] < 0 or ResidGrad[Index] < 0):
            Passed[Index] = True;
            Index = Index - 1;

        # Loop ahead to find the extent of the trough
        Index = i + 1;
        while Index < Residuals.shape[0] and (not Passed[Index]) and (Residuals[Index] < 0 or ResidGrad[Index] > 0):
            Passed[Index] = True;
            Index = Index + 1;
    return Passed

def savgol_filter_padding(signal,SG_flen,SG_ord):
    """
    Wrapper function for using savgol filter.
    Meant to avoid using a filter length larger than the length of the signal.
    """
    if SG_flen == 0:
        SG_flen = int(3600 / 4) + 1
        if len(signal) < SG_flen:
            SG_flen = len(signal) - 1
        if SG_flen % 2 == 0:
            SG_flen = SG_flen - 1

    start=0
    end=len(signal)

    if len(signal) <= SG_flen:
        signal =np.pad(signal, SG_flen, mode='reflect',reflect_type='odd')
        start = SG_flen
        end=len(signal)-SG_flen

    return savgol_filter(signal, SG_flen, SG_ord)[start:end]

def get_pressure_tube_surface(frame_of_lags, ro_start, ro_end, SG_flen=300, SG_ord=5):
    """
    Smooths the lags to get the general trend
    """
    indications = np.zeros(frame_of_lags.shape)
    indications[ro_start:ro_end+ 1] = True
    indication_removed = np.array(interpolate_indications(frame_of_lags, indications))

    # smoothed signal without indications
    return savgol_filter_padding(indication_removed, SG_flen, SG_ord)