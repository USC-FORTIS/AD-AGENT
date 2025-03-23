class CodeQuality:
    def __init__(self,code,algorithm,error_message,detected_anomalies,true_anomalies,review_count):
        self.code = code
        self.algorithm = algorithm
        self.error_message = error_message
        self.detected_anomalies = detected_anomalies
        self.true_anomalies = true_anomalies
        self.review_count = review_count
        
    