import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve





def plot_calibration_curve(y_true, pred_proba, n_bins=10):

	fraction_of_positives, mean_predicted_value = calibration_curve(y_true, 
	                                                            pred_proba, 
	                                                            normalize=False, 
	                                                            n_bins=10)
	plt.plot([0,1],[0,1], 'k:', label='Perfectly Calibrated')
	plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Actual")
	
	plt.ylabel("Fraction of positives")
	plt.title('Calibration plots  (reliability curve)')