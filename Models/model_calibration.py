from sklearn.calibration import CalibratedClassifierCV


def calibrate(X_val, y_val, estimator):

    clf = CalibratedClassifierCV(base_estimator=estimator, 
                                method='isotonic', cv='prefit')

    clf.fit(X_val, y_val)
    return clf

def main():
    pass    


if __name__ == '__main__':
    main()