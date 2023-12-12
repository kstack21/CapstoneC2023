# Predicting Thrombosis in Patients with Peripheral Artery Disease 

Northeastern Capstone Group C 2023

Design Team: Luis Arroyo, Andrea Chávez Muñoz, Luca Peirce, Kayley Stack, and Faith Toner 

Design Advisor: Dr. Anahita Dua 

Design Sponsor: Dr. Anahita Dua 


## Motivation, Background, and Overview 

Peripheral Artery Disease (PAD) is a form of cardiovascular disease that often occurs due to the buildup of atherosclerotic plaque in the blood vessels of the lower extremities, which results in reduced blood flow. This can lead to a myriad of problems, including pain in the lower extremities, minimization or loss of extremity function, or ultimately amputation of the affected limb if left untreated. A common treatment for PAD involves surgery on the affected artery. There are several types of surgery to treat PAD, including bypass surgery, angioplasty, and/or stent placement. These treatments come with risks, however, with the biggest concern being the occurrence of a thrombotic event, or a blood clot forming. This clot restricts blood flow, and if dislodged could travel through the blood vessels to the brain, heart, or lungs, causing more serious problems. Even without traveling, the blockage of arteries due to thrombosis leads to lower extremity amputations for many elderly patients after undergoing a revascularization procedure. Currently, the way that physicians attempt to prevent post-op thrombosis consists of a standardized prescription of blood thinners to patients, regardless of their own personal risk factors. This “one size fits all” approach poses multiple problems, and the appeal of a tool to predict the risk of thrombosis for an individual patient became apparent following a discussion with Dr. Anahita Dua, a vascular surgeon at Massachusetts General Hospital (MGH). A Streamlit app capable of seamlessly generating XGBoost machine learning algorithms was developed. The app is crafted with clear and simple instructions, providing users with a guided experience in creating and utilizing a predictive model for thrombotic event prediction in patients. Recognizing the intricacies of blood-related data, particular emphasis was placed on addressing the high correlation among TEG results. The overarching objective was to bridge the gap between physicians and data science, enabling healthcare professionals with limited machine learning or coding expertise to effortlessly leverage the algorithm. This not only facilitates immediate use but also empowers physicians to independently train new models as additional data becomes available. 

 

## Key Design Requirements 

DRQ-001: All patient data shall be managed in a HIPAA-compatible way. 

DRQ-002: The app shall provide a user-friendly platform to assist in creating a model and determining a patient’s risk score. 

DRQ-003: Each time the app creates a new model, the model shall be validated using industry accepted techniques of machine learning methods. 

DRQ-004: The app shall outline the biggest contributors to a patient's risk score. 

 

## Design Solution/Procedure Summary 

## Predictive Model 

The predictive model is generated using an XGBoost machine learning algorithm. This algorithm was chosen after a thorough investigation of 12 possible algorithms. Factors such as number of parameters, noise resistance, overfitting resistance, and intrinsic feature selection were all considered and resulted in random forest, gradient boost, and XGBoost being the top three algorithms considered. XGBoost is a specific implementation of gradient boosting with several enhancements, making it efficient and resilient to missing data values. XGBoost was chosen because the MGH dataset, as well as potential future datasets, contains numerous missing values due to patients not attending follow-up appointments. 

 

## User Interface 

The user interface was designed in Python using Streamlit to create a desktop application. The first page welcomes the user to the application and provides an overview of how to navigate the app. 

On the next page, called “Predict Thrombotic Risk”, users can upload data for a single patient along with a predictive model. After uploading the patient's data through an Excel file, a summary of their basic information is presented to ensure that the correct patient has been uploaded. Then, a plot shows a summary of the data used to train the model and the model validity scores, enabling users to compare the training data with the patient's information and assess its impact on the model's predictions. Finally, the patient's risk of thrombosis is displayed, considering both their TEG results and baseline comorbidities. 

The model training page allows users to train and download a new predictive model for future use on the "Train Model" page. To begin, users upload a dataset containing information from multiple patients, containing their baseline comorbidities, TEG values, and associated events. The app then generates an initial model based on the provided baseline and TEG results. From there, the importance of each parameter is calculated using Shapely values. The users are then given the flexibility to select the parameters they wish to include in the second model, mitigating the influence of collinear factors that may be interrelated. Upon making their selections, users can initiate the second model generation by clicking the "Train and Validate" button. After the model is created, users have the option to download either the first model, which includes all parameters, or the second model, which excludes certain parameters that might be correlated given the nature of blood samples. 
