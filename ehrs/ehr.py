class EHR(object):
    def __init__(self):
        super().__init__()
    
        self.icustays = None
        self.patients = None
        self.admissions = None
        self.diagnoses = None

    @property
    def icustays(self):
        return self.icustays
    
    @property
    def patients(self):
        return self.patients
    
    @property
    def admissions(self):
        return self.admissions
    
    @property
    def diagnoses(self):
        return self.diagnoses