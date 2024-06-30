# Disease-Prediction-from-Medical-Data
import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import csv
import joblib
#knn_from_joblib.predict(X_test) 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

with open('../Medical_dataset/intents_short.json', 'r') as f:
    intents = json.load(f)

intents
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()
knn= joblib.load('../model/knn.pkl')  

# preprocess sentence
def preprocess_sent(sent):
    t=nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w.lower()) for w in t if (w not in set(stopwords.words('english')) and w.isalpha())])

# BOW of prepocessed sentence
def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# predict possible symptom in a sentence
def predictSym(sym,vocab,app_tag):
    sym=preprocess_sent(sym)
    bow=np.array(bag_of_words(sym,vocab))
    res=cosine_similarity(bow.reshape((1, -1)), df).reshape(-1)
    order=np.argsort(res)[::-1].tolist()
    possym=[]
    for i in order:
        if app_tag[i].replace('_',' ') in sym:
            return app_tag[i],1
        if app_tag[i] not in possym and res[i]!=0:
            possym.append(app_tag[i])
    return possym,0

# input : patient symptoms / output : OHV DataFrame 
def OHV(cl_sym,all_sym):
    l=np.zeros([1,len(all_sym)])
    for sym in cl_sym:
        l[0,all_sym.index(sym)]=1
    return pd.DataFrame(l, columns =all_symp)

def contains(small, big):
    a=True
    for i in small:
        if i not in big:
            a=False
    return a

# returns possible diseases 
def possible_diseases(l):
    poss_dis=[]
    for dis in set(disease):
        if contains(l,symVONdisease(df_tr,dis)):
            poss_dis.append(dis)
    return poss_dis

# input: Disease / output: all symptoms
def symVONdisease(df,disease):
    ddf=df[df.prognosis==disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()

# preprocess symptoms    
def clean_symp(sym):
    return sym.replace('_',' ').replace('.1','').replace('(typhos)','').replace('yellowish','yellow').replace('yellowing','yellow')     

def getInfo():
    # name=input("Name:")
    print("Your Name \n\t\t\t\t\t\t",end="=>")
    name=input("")
    print("hello ",name)
    return str(name)


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

def getDescription():
    global description_list
    with open('../Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('../Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('../Medical_dataset/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def calc_condition(exp,days):
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp))>13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")

getSeverityDict()
getprecautionDict()
getDescription()
# read TF IDF symptoms and Training diseases
df=pd.read_csv('../Medical_dataset/tfidfsymptoms.csv')
df_tr=pd.read_csv('../Medical_dataset/Training.csv')
vocab=list(df.columns)
disease=df_tr.iloc[:,-1].to_list()
all_symp_col=list(df_tr.columns[:-1])
all_symp=[clean_symp(sym) for sym in (all_symp_col)]
app_tag=[]
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        app_tag.append(tag)


def main_sp(name):
    #main Idea: At least two initial sympts to start with

    #get the 1st syp ->> process it ->> check_pattern ->>> get the appropriate one (if check_pattern==1 == similar syntaxic symp found)
    print("Hi Mr/Ms "+name+", can you describe you main symptom ?  \n\t\t\t\t\t\t",end="=>")
    sym1 = input("")
    psym1,find=predictSym(sym1,vocab,app_tag)
    if find==1:
        sym1=psym1
    else:
        i=0
        while True and i<len(psym1):
            print('Do you experience '+psym1[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym1=psym1[i]
                break
            else:
                i=i+1

    print("Is there any other symtom Mr/Ms "+name+"  \n\t\t\t\t\t\t",end="=>")
    sym2=input("")
    psym2,find=predictSym(sym2,vocab,app_tag)
    if find==1:
        sym2=psym2
    else:
        i=0
        while True and i<len(psym2):
            print('Do you experience '+psym2[i].replace('_',' '))
            rep=input("")
            if str(rep)=='yes':
                sym2=psym2[i]
                break
            else:
                i=i+1

    #create patient symp list
    all_sym=[sym1,sym2]
    #predict possible diseases
    diseases=possible_diseases(all_sym)
    stop=False
    print("Are you experiencing any ")
    for dis in diseases:
        if stop==False:
            for sym in symVONdisease(df_tr,dis):
                if sym not in all_sym:
                    print(clean_symp(sym)+' ?')
                    while True:
                        inp=input("")
                        if(inp=="yes" or inp=="no"):
                            break
                        else:
                            print("provide proper answers i.e. (yes/no) : ",end="")
                    if inp=="yes":
                        all_sym.append(sym)
                        dise=possible_diseases(all_sym)
                        if len(dise)==1:
                            stop=True 
                            break
                    else:
                        continue
    return knn.predict(OHV(all_sym,all_symp_col)),all_sym

def chat_sp():
    a=True
    while a:
        name=getInfo()
        result,sym=main_sp(name)
        if result == None :
            ans3=input("can you specify more what you feel or tap q to stop the conversation")
            if ans3=="q":
                a=False
            else:
                continue

        else:
            print("you may have "+result[0])
            print(description_list[result[0]])
            an=input("how many day do you feel those symptoms ?")
            if calc_condition(sym,int(an))==1:
                print("you should take the consultation from doctor")
            else : 
                print('Take following precautions : ')
                for e in precautionDictionary[result[0]]:
                    print(e)
            print("do you need another medical consultation (yes or no)? ")
            ans=input()
            if ans!="yes":
                a=False
                print("!!!!! thanks for using ower application !!!!!! ")

if __name__=='__main__':
    chat_sp()
 2,597 changes: 2,597 additions & 0 deletions2,597  
Draft/predictSymp.ipynb
Large diffs are not rendered by default.

 1,442 changes: 1,442 additions & 0 deletions1,442  
Draft/preprocess 1.ipynb
Large diffs are not rendered by default.

 934 changes: 934 additions & 0 deletions934  
Draft/word2vec Disease.ipynb
Large diffs are not rendered by default.

 150 changes: 150 additions & 0 deletions150  
Medical_dataset/DiseaseUMLS.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,150 @@
UMLS,CUI
hypertensive disease,C0020538
diabetes,C0011847
depression mental,C0011570
coronary arteriosclerosis,C0010054
pneumonia,C0032285
failure heart congestive,C0018802
accident cerebrovascular,C0038454
asthma,C0004096
myocardial infarction,C0027051
hypercholesterolemia,C0020443
infection,C0021311
infection urinary tract,C0042029
anemia,C0002871
chronic obstructive airway disease,C0024117
dementia,C0497327
insufficiency renal,C1565489
confusion,C0009676
degenerative polyarthritis,C0029408
hypothyroidism,C0020676
anxiety state,C0700613
malignant neoplasms,C0006826
acquired immuno-deficiency syndrome,C0001175
cellulitis,C0007642
gastroesophageal reflux disease,C0017168
septicemia,C0036690
deep vein thrombosis,C0149871
dehydration,C0011175
neoplasm,C0027651
embolism pulmonary,C0034065
epilepsy,C0014544
cardiomyopathy,C0878544
chronic kidney failure,C0022661
carcinoma,C0007097
hepatitis C,C0019196
peripheral vascular disease,C0085096
psychotic disorder,C0033975
hyperlipidemia,C0020473
bipolar disorder,C0005586
obesity,C0028754
ischemia,C0022116
cirrhosis,C1623038
exanthema,C0015230
benign prostatic hypertrophy,C0005001
kidney failure acute,C0022660
mitral valve insufficiency,C0026266
arthritis,C0003864
bronchitis,C0006277
hemiparesis,C0018989
osteoporosis,C0029456
transient ischemic attack,C0007787
adenocarcinoma,C0001418
paranoia,C1456784
pancreatitis,C0030305
incontinence,C0021167
paroxysmal dyspnea,C0013405
hernia,C0019270
malignant neoplasm of prostate,C0376358
edema pulmonary,C0034063
lymphatic diseases,C0024228
stenosis aortic valve,C0003507
malignant neoplasm of breast,C0006142
schizophrenia,C0036341
diverticulitis,C0012813
overload fluid,C0546817
ulcer peptic,C0030920
osteomyelitis,C0029443
gastritis,C0017152
bacteremia,C0004610
failure kidney,C0035078
sickle cell anemia,C0002895
failure heart,C0018801
upper respiratory infection,C0041912
hepatitis,C0019158
hypertension pulmonary,C0020542
deglutition disorder,C0011168
gout,C0018099
thrombocytopaenia,C0040034
hypoglycemia,C0020615
pneumonia aspiration,C0032290
colitis,C0009319
diverticulosis,C1510475
suicide attempt,C0038663
Pneumocystis carinii pneumonia,C0032305
hepatitis B,C0019163
parkinson disease,C0030567
lymphoma,C0024299
hyperglycemia,C0020456
encephalopathy,C0085584
tricuspid valve insufficiency,C0040961
Alzheimer's disease,C0002395
candidiasis,C0006840
neuropathy,C0442874
kidney disease,C0022658
fibroid tumor,C0023267
glaucoma,C0017601
neoplasm metastasis,C0027627
malignant tumor of colon,C0007102
ketoacidosis diabetic,C0011880
tonic-clonic epilepsy,C0014549
malignant neoplasms,C0006826
respiratory failure,C1145670
melanoma,C0025202
gastroenteritis,C0017160
malignant neoplasm of lung,C0242379
manic disorder,C0024713
personality disorder,C0031212
primary carcinoma of the liver cells,C0019204
emphysema pulmonary,C0034067
hemorrhoids,C0019112
spasm bronchial,C0006266
aphasia,C0003537
obesity morbid,C0028756
pyelonephritis,C0034186
endocarditis,C0014118
effusion pericardial,C0031039
chronic alcoholic intoxication,C0001973
pneumothorax,C0032326
delirium,C0011206
neutropenia,C0027947
hyperbilirubinemia,C0020433
influenza,C0021400
dependence,C0439857
thrombus,C0087086
cholecystitis,C0008325
hernia hiatal,C0019291
migraine disorders,C0149931
pancytopenia,C0030312
cholelithiasis,C0008350
tachycardia sinus,C0039239
ileus,C1258215
adhesion,C0001511
delusion,C0011253
affect labile,C0233472
decubitus ulcer,C0011127
depressive disorder,C0011581
coronary heart disease,C0010068
primary malignant neoplasm,C1306459
HIV,C0019682
systemic infection,C0243026
carcinoma prostate,C0600139
carcinoma breast,C0678222
oralcandidiasis,C0006849
carcinoma colon,C0699790
tonic-clonic seizures,C0494475
carcinoma of lung,C0684249
pericardial effusion body substance,C1253937
biliary calculus,C0242216
hiv infections,C0019693
sepsis (invertebrate),C1090821
 313 changes: 313 additions & 0 deletions313  
Medical_dataset/DiseaseUMLS1.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,313 @@
Disease_CUI,Disease
C0162565,"porphyria, acute intermittent"
C0162566,porphyria cutanea tarda
C0004238,atrial fibrillation
C0023212,left heart failure
C0235527,right heart failure
C0018801,heart failure
C0007194,hypertrophic cardiomyopathy
C0007196,"cardiomyopathy, restrictive"
C0007193,dilated cardiomyopathy
C0024790,paroxysmal nocturnal hemoglobinuria
C0037889,anemia globular cell
C1268964,myelodysplastic syndrome not classifiable
C0947845,acute and chronic sarcoid
C0340164,loefgren syndrome
C0042171,heerfordt syndrome
C0085278,antiphospholipid syndrome
C0034155,"microangiopathy, thrombotic"
C0020532,hypersplenism
C0032227,pleural effusion
C0034063,pulmonary edema
C0032326,pneumothorax
C0877430,chronic asthma
C0024117,copd
C0032285,pneumonia
C0034067,emphysema
C0034069,pulmonary fibrosis
C0034065,pulmonary embolism
C0149514,acute bronchitis
C0006271,bronchiolitis
C0343528,pontiac - fever
C0023241,pneumonia due to legionella
C0857946,atypical pneumonia
C1535939,pneumocystis jirovecii - pneumonia
C0041296,tuberculosis
C0041333,"tuberculosis, urogenital"
C1740806,atypical mycobacteriosis
C0027145,myxoedem
C0002390,"alveolitis, exogenous - allergic"
C0002390,exogenous - allergic alveolitis
C0043167,pertussis
C0017150,gastrinoma
C0085695,chronic gastritis
C0023788,whipple's disease
C0235950,zinc deficiency
C0024473,magnesium deficiency
C0032827,potassium deficiency
C0042842,vitamin - a deficiency
C0007570,gluten - sensitive enteropathy
C0949570,"allergy, wheat"
C0221036,acrodermatitis enteropathica
C0017155,menetrian disease
C0023530,leukocytopenia
C0221406,cushing disease
C0020461,hypercalemia
C0010068,coronary heart disease
C0029456,osteoporosis
C1258215,ileus
C0023890,cirrhosis of the liver
C0018378,guillain barre syndrome
C0019937,horner syndrome
C0338481,eye migraines
C0010346,crohn's disease
C0026764,plasma cell myeloma
C1527249,colorectal cancer
C0026896,myasthenia gravis
C0003873,"arthritis, rheumatoid"
C0002871,anemia
C0039070,syncope
C0026769,multiple sclerosis
C0020621,hypocalemia
C1306557,chronic venous insufficiency
C0008350,cholelithiasis
C0020550,hyperthyroidism
C0025289,meningitis
C0022661,chronic kidney failure
C0027726,nephrotic syndrome
C0041948,uraemia
C0020676,hypothyroidism
C0235295,Acute abdomen
C0020542,pulmonary hypertension
C3495801,granulomatosis with polyangiitis
C0020615,hypoglycemia
C0011570,depression
C0003962,ascites
C0031511,phaeochromocytoma
C0022346,icterus
C0017658,glomerulonephritis
C0020649,hypotension
C0340643,aortic dissection
C0040425,tonsillitis
C0036262,scabies
C0854330,herpes zoster dermatitis
C1704272,benign prostatic hyperplasia
C0235974,pancreatic cancer
C0011848,diabetes insipidus
C0020437,hypercalcaemia
C0021345,Pfeiffer fever
C0162316,iron deficiency anemia
C0852445,hodgkin's disease
C0035012,Reiter's disease
C0036974,shock
C0010674,cystic fibrosis
C0014118,endocarditis
C0034734,Raynaud
C0036341,schizophrenia
C0038525,subarachnoid hemorrhage
C0019187,"hepatitis, alcoholic"
C0032461,polycythemia
C0004943,behcet's disease
C0011206,delirium
C0020651,orthostatic syndrome
C0011633,dermatomyositis
C0002895,sickle cell anemia
C0018995,haemochromatosis
C0019151,"encephalopathy, hepatic"
C0002726,amyloidosis
C0035222,acute respiratory distress syndrome
C1144799,hypertensive cardiomyopathy
C0011175,desiccosis
C1384514,primary hyperaldosteronism
C0034072,cor pulmonale
C0017168,gastroesophageal reflux disease
C0033860,psoriasis
C1145670,respiratory failure
C0549346,ischialgia
C0043121,wernicke - encephalopathy
C0019202,wilson's disease
C0012813,diverticulitis
C0025281,meniere's disease
C0024305,non-hodgkin lymphoma
C0086543,cataract
C0014038,encephalitis
C0024530,malaria
C0034186,pyelonephritis
C0032026,pityriasis rosea
C0017551,gilbert syndrome
C0600139,prostate cancer
C0013080,down syndrom
C0019158,hepatitis
C0001824,agranulocytosis
C0155502,benign paroxysmal positional vertigo
C0023434,chronic lymphocytic leukemia
C0003504,aortic regurgitation
C0149931,migraine
C0007787,Transient ischemic attack
C0023473,chronic myeloid leukemia
C0006267,bronchiectasis
C0024141,systemic lupus erythematosus
C0040127,thyrotoxic crisis
C1527311,cerebral edema
C0019112,hemorrhoids
C0038358,gastric ulcer
C0221757,alpha - 1 proteinase inhibitor deficiency
C0010481,cushing syndrome
C0029925,ovarian cancer
C0002874,aplastic anemia
C0018081,gonorrhea
C0039232,av node - reentry - tachycardia
C0008049,varicella
C0006285,bronchopneumonia
C0221069,anterior spinal artery syndrome
C0022972,lambert - eaton syndrome
C0016053,fibromyalgia
C0007398,catatonia
C0021141,Schwartz - Bartter Syndrome
C0042974,von - willebrand - disease
C0023176,lead poisoning
C0302592,cervical cancer
C1306571,liver failure
C0012546,diphtheria
C0001206,acromegaly
C0085278,antiphospholipid syndrome
C0020428,hyperaldosteronism
C0003864,arthritis
C0007115,thyroid cancer
C0151945,cerebral venous thrombosis
C0542142,paralysis of the recurrent laryngeal nerve
C0007134,kidney cell carcinoma
C0451641,urolithiasis
C0021400,influenza
C0520764,clostridia - gastroenteritis
C0035400,reye syndrome
C0032708,porphyria
C0376175,facial palsy
C0026691,kawasaki syndrome
C0011881,diabetic nephropathy
C0036285,Scarlet fever
C0017152,gastritis
C0264558,tension pneumothorax
C0008373,cholesteatoma
C0031046,pericarditis
C0020258,"hydrocephalus, normal pressure"
C0152018,esophageal carcinoma
C2239176,hepatocellular carcinoma
C0040028,essential thrombocythaemia
C0751761,upper respiratory tract sleep apnea syndrome
C0008677,Chronic bronchitis
C0153633,brain tumor malignant
C0038531,subclavian - steal syndrome
C0027059,myocarditis
C0021807,intertrigo
C0008311,cholangitis
C0566602,primary sclerosing cholangitis
C0024586,carcinoid syndrome
C0852437,myeloproliferative diseases (excluding leukaemias)
C0149518,acute gastritis
C0035579,rickets
C3888977,uterine myomatosis
C0023467,acute myeloid leukemia
C0029882,otitis media
C0006107,commotio cerebri
C0014869,reflux esophagitis
C0010692,cystitis
C1510475,diverticulosis
C0011119,diving disease
C0752347,lewy - body - dementia
C0149801,urosepsis
C0043019,Wallenberg Syndrome
C0558355,Tonsil cancer
C0206702,klatskin tumor
C0024225,lymphangitis
C0030920,"ulcer, gastroduodenal"
C0017601,glaucoma
C3495928,thyroid-related orbitopathy
C0949690,spondylarthritis
C0003492,coarctation of the aorta
C0152032,urinary retention
C0027849,Neuroleptic Malignant Syndrome
C0030353,congestive papilla
C0392549,infantile cerebral palsy
C0035305,retinal reading
C0041976,urethritis
C0019069,haemophilia a
C0017205,gaucher disease
C1704436,peripheral arterial disease
C0035410,rhabdomyolysis
C0035920,rubella
C0023449,acute lymphoblastic leukemia
C0014752,erythrasma
C0028734,nycturia
C0948968,Osteomyelofibrosis
C0002390,"allergic alveolitis, exogenous"
C0003550,broca - aphasia
C0024904,mastoiditis
C0039621,tetanie
C0259756,herxheimer - reaction
C1258104,"sclerosis, progressive systemic"
C0030436,parakeratosis
C0087031,"Still disease, juvenile form"
C0175702,williams - beuren - syndrome
C0340164,loefgren syndrome
C0020305,hydrops fetalis
C0007194,obstructive cardiomyopathy
C0553720,sphaerocytosis
C0017920,glucose phosphatase deficiency 06
C0002895,sickle cell disease
C0002874,anemia aplastic
C0023443,hair cell leukemia
C0036920,sezary syndrome
C0085669,acute leukemia
C0002986,Fabry Syndrome
C0398650,immune thrombocytopenic purpura
C0014121,infectious endocarditis
C0035436,rheumatic fever
C0026269,mitral valve stenosis
C0026266,mitral regurgitation
C0026267,mitral valve prolapse
C0003507,aortic stenosis
C0003504,aortic regurgitation
C1956257,pulmonary stenosis
C0018817,atrial septal defect
C0018818,ventricular septal defect
C0013481,ebstein - anomaly
C0039685,fallot tetralogy
C0024796,marfan's syndrome
C0151595,digitalis intoxication
C0041234,chagas - disease
C0007177,heart bag tamponade
C0265144,chronic constrictive pericarditis
C0027051,Heart attack
C0024117,chronic obstructive pulmonary disease
C0032269,pneumococcal infections
C0026936,"mycoplasma infection, unspecified"
C1096584,chlamydia pneumoniae - infection
C0029291,ornithosis
C0034362,q - fever
C0003175,infection with bacillus anthracis
C0035235,respiratory syncytial virus infections
C0085740,mendelson's syndrome
C0037116,silicosis
C0007121,bronchial carcinoma
C0036202,sarcoid
C1321756,achalasia
C0014854,esophageal diverticulum
C0024623,gastric cancer
C0016470,food allergy
C0302858,protein loss syndrome
C0009324,ulcerative colitis
C0400821,microscopic colitis
C0022104,irritable bowel
C0001339,acute pancreatitis
C0149521,chronic pancreatitis
C0206695,"carcinoma, neuroendocrine"
C0021670,insulinoma
C0086768,verner morrison syndrome
C0019189,"hepatitis, chronic"
C0023896,alcoholic liver disease
C0162557,"liver failure, acute"
C3805004,post-infectious glomerulonephritis
C0451728,rapid - progressive nephritic syndrome
C3805154,renal osteopathy syndrome
C0027708,nephroblastoma
 47 changes: 47 additions & 0 deletions47  
Medical_dataset/NEWTEST.csv
Large diffs are not rendered by default.

 5,054 changes: 5,054 additions & 0 deletions5,054  
Medical_dataset/NEWTRAIN.csv
Large diffs are not rendered by default.

 5,054 changes: 5,054 additions & 0 deletions5,054  
Medical_dataset/NEW_DS.csv
Large diffs are not rendered by default.

 410 changes: 410 additions & 0 deletions410  
Medical_dataset/SymptomUMLS(1).csv
Large diffs are not rendered by default.

 374 changes: 374 additions & 0 deletions374  
Medical_dataset/SymptomUMLS(2).csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,374 @@
Symptom_CUI,SymptomUMLS
C0702266,basophilia
C0041834,redness
C0014804,erythromelalgia
C0014591,epistaxis
C0406191,pseudofolliculitis
C0042165,anterior uveitis
C0042167,posterior uveitis
C0157738,chronic ulcer skin (incl varikoes)
C0233407,disorientation
C0392156,akathisia
C0085631,agitation
C0015967,hyperthermia
C0027121,myositis
C0011633,dermatomyositis
C0027059,myocarditis
C0549493,alveolitis
C0041834,erythema
C0085642,livedo reticularis
C0026846,"atrophy, muscle"
C0006664,calcinosis cutis
C0242184,hypoxia
C0011175,dehydration
C0001122,acidosis
C0033802,pseudogout
C0009676,confusion
C0013144,sleepiness
C0232766,flutter tremor
C1142109,sopor
C0009677,macroglossia
C0020578,hyperventilation
C0002064,"alkalosis, respiratory"
C0035229,respiratory failure
C0020440,hypercapnia
C0001127,respiratory acidosis
C0035423,rheology
C0220983,metabolic alkalosis
C0011168,difficulties swallowing
C2004489,regurgitation
C0018834,heartburn
C0035339,retinoids
C0239161,dactylitis
C0019087,Bleeding tendency
C0947757,lymphoma ane
C0427544,monocytopenia
C1142446,bicytopenia
C0858867,reticulocytopenia
C0085662,macrocytosis
C0221281,poikilocytosis
C0333942,polychromatism
C0221278,anisocytosis
C1868945,polyglobulia
C0234979,dysdiadochokinesis
C0193388,liver biopsy
C0042571,vertigo
C0494562,other hearing loss
C1136085,"gammopathy, monoclonal"
C0001824,granulocytopenia
C0086438,hypogammaglobulinemia
C0525027,capsulorhexis
C1956311,phacectomy
C0085636,photosensitivity
C0020615,hypoglycemia
C0036974,circulatory shock
C0035078,kidney failure
C0085593,chills
C0027497,nausea
C0919671,subileus
C0040252,tinea corporis
C0013595,eczema
C0039128,syphilis
C0033860,psoriasis
C0085657,pityriasis alba
C0478132,other micturition disorders
C0149707,haematospermia
C0265610,clinodactyly
C0234233,tenderness
C0015230,exanthem
C0038363,aphthous stomatitis
C1739421,acute vertigo
C0240595,rotating nystagmus
C0026470,mgus
C0263338,chronic urticaria
C0854330,herpes zoster dermatitis
C0026946,mycoses
C0038218,asthma attack
C0023212,left heart failure
C0236018,aura
C2363911,headache under physical strain
C0751466,phonophobia
C0085636,photophobia
C0003537,aphasia
C0260662,hearing disruptions
C0852418,speech disorders
C0005699,blast thrust
C0920063,uric acid high
C0024110,lung abscess nnb
C0221244,dandruff
C0024143,lupus nephritis
C0002880,autoimmune hemolytic anemia
C0221201,maculoeses exanthem
C0235499,multiple telangiectasias
C0852968,alopecia effluvium
C0033975,psychoses
C0007971,cheilitis
C0014868,oesophagitis
C0016382,flush
C4082299,bulbar paralysis
C0011175,desiccosis
C0016167,anal fissure
C2363872,marisks
C0033775,pruritus ani
C0034888,rectal prolapse
C0009402,colorectal cancer
C0001883,airway obstruction nnb
C0034067,emphysema
C0022354,cholestatic icterus
C0438237,liver enzymes abnormal
C0020488,hypernatriaemia
C0546817,hypervolaemia
C0013390,dysmenorrhea
C0016205,flatulence / meteorism / belching
C0027947,neutropenia
C0014591,nosebleeds
C0017574,gingivitis
C0243026,sepsis
C0425945,menstruation lengthened
C0149745,mouth ulcers
C0040425,tonsillitis
C0007642,phlegmon
C0023531,leukoplakia
C0007860,cervicitis
C0004769,bartholinitis
C0033246,proctitis
C0040264,ear noises
C0007758,cerebellar ataxia
C0020458,hyperhidrosis
C0423600,globe feeling of the pharynx
C0085628,stupor
C0233612,flexibilitas cerea
C0026884,mutism
C0013528,echolalia
C0026821,muscle cramps
C0240735,personality change
C0020625,hyponatriaemia
C0040440,tooth extraction
C0025323,menorrhagia
C0018924,hemarthrosis
C0544864,lead edge
C0085584,encephalopathy
C0016330,fluorine
C0012556,pharyngeal diphtheria
C0032541,polyneuritis
C0456909,vision loss
C0036572,seizure
C0149931,migraine
C0238457,renal vein thrombosis
C0034734,raynaud
C0027051,myocardial infarction
C0013922,embolism
C0000809,"abort, more habitual"
C0043251,trauma
C0035243,respiratory infection
C0014544,epileptical attack
C0751495,focal attacks
C0038450,stridor
C0042341,varicocele
C0016199,flank pain
C0239937,microhaematuria
C0152169,ureteral colic
C1963891,gas fire
C0031154,peritonitis
C0016479,food poisoning
C0520459,necrotizing enterocolitis
C0006057,botulism
C0039614,tetanus
C1257843,pseudomembranous colitis
C0086565,liver dysfunction
C0852553,psychiatric symptoms ane
C0221328,nasolabial fold
C0224176,platysma
C0014034,enanthem
C0241262,raspberry tongue
C0001925,albuminuria
C0020473,hyperlipidaemia
C0013456,ear pain
C0018777,conduction impairment
C0575081,gait disorders
C0042024,urinary incontinence
C1112468,small step
C0025222,melaena
C0018926,haematemesis
C0267072,oesophageal dysphagia
C0037036,hypersalivation
C0155773,portal vein thrombosis
C0013911,emaciation
C0017565,bleeding gums
C0019087,hemorrhagic diathesis
C0740394,hyperuricaemia
C0026635,mouth breathing
C0037384,snoring
C0004339,auscultation
C0085624,burn
C0008370,cholestasis
C0016059,fibrosis
C0025222,tarry stool
C0085577,normocytic anemia
C0014724,belching
C0392452,craniotabes
C0426824,rachitic rosary
C0011334,Caries
C0005682,bladder
C0000727,Acute abdomen
C0029883,middle ear effusion
C0549099,perforation
C0086625,facial muscles
C0037822,speech impairment
C0260662,hearing
C0520546,microembolism of the lungs
C0162297,Apnea
C0478875,other falls on the same level
C1527344,dysphonia
C1384666,hypacusis
C0231218,feeling sick
C0034941,pupil reflex
C0235267,Red eye
C0423602,foreign body feeling
C0242490,enthesopathy
C0375553,overflow incontinence
C0201973,ck
C0013384,dyskinetic syndrome
C0016242,mouches floaters
C0852361,skin bleeding
C0015934,fetal growth retardation
C0027080,myoglobinuria
C0025007,measles
C0015231,exanthema subitum
C0006840,candidose
C0011616,contact dermatitis
C0036508,"dermatitis, seborrheic"
C0010692,cystitis
C0042024,bladder incontinence
C1704272,benign prostatic hyperplasia
C0032961,pregnancy
C0012798,diuretics
C0234488,paraphasia
C0013456,earache
C0155540,otorrhea
C0085616,vasospasm
C0150988,sclerodactyly
C0026034,microstomy
C0021847,intestinal pseudo-obstruction
C0151654,myocardial fibrosis
C0035085,kidney infarction
C0155765,microangiopathy
C0478124,other skin changes
C0277799,intermittent fever
C0036749,serositis
C0495672,"hepatomegaly, not elsewhere classified"
C0495673,"splenomegaly, not elsewhere classified"
C0235999,pain in the neck / shoulder
C0743912,ferritin high
C1141969,occupational therapy
C0020224,polyhydramnios
C0085612,ventricular arrhythmia
C0042514,tachycardia ventricular
C0235983,normochromic anemia
C0206160,reticulocytosis
C0038454,apoplectic insult
C0575805,hand swelling
C0029443,osteomyelitis
C0011606,erythroderma
C0870082,hyperkeratosis
C0221260,onychodystrophy
C0012739,consumption coagulopathy
C0948840,meningeosis leucaemica
C1706559,cornea verticillata
C0007787,tia
C0005959,bone marrow hyperplasia
C1142004,myocardial abscess
C0023211,left bundle branch block
C0264686,coronary embolism
C0149651,watch glass nails
C0018989,hemiparesis
C0027697,nephritis
C0234906,erythema annulare
C0030270,pancarditis
C0004238,atrial fibrillation
C0020542,pulmonary hypertension
C0332610,facies mitralis
C0003811,arrhythmia
C0036980,cardiogenic shock
C0151636,ventricular extrasystoles
C0857087,dizzy spells
C0002962,stenocardia
C0235527,right heart failure
C0424605,development delay
C0948783,bronchopulmonary infection
C0002962,heartache
C0857375,hypoxic standstill
C0158731,keel breast
C0016842,funnel breast
C0016202,buckle foot
C0032326,pneumothorax
C0409495,protrusio acetabuli
C0022821,kyphosis
C0221358,dolichocephaly
C3494422,retrognathy
C0152459,striae
C0027092,myopia
C0026267,mitral valve prolapse
C0003486,aortic aneurysm
C0265004,aortic ectasia
C0003706,arachnodactyly
C0392477,"flat foot, congenital"
C0026266,mitral regurgitation
C0085610,sinus bradycardia
C0030587,paroxysmal atrial tachycardia
C0340464,extrasystole
C0235477,bigeminus
C0264886,conduction disorders
C0023212,left heart failure
C0085298,sudden heart death
C0192389,esophageal dilatation
C0025164,megaoesophagus
C0025160,megacolon
C1168392,mean venous pressure increases
C0232118,kiss mouth positive
C0020532,hypersplenism
C0600177,low cardiac output syndrome
C0042510,ventricular fibrillation
C0025289,meningitis
C0010043,corneal ulcer
C0040586,tracheobronchitis
C0206061,interstitial pneumonia
C0006277,bronchitis
C0006285,bronchopneumonia
C0043167,whooping cough
C0006266,bronchospasm
C0392315,glottic spasm
C0159054,sputum abnormal
C0542142,recurrent palsy
C0427855,pleural exudate
C0344306,intercostal neuralgia
C0085655,polymyositis
C0836924,thrombocytosis
C0263661,osteoarthropathy
C0019079,haemoptoe
C0239134,cough with sputum
C0700198,aspiration
C0013395,dyspepsia
C0017181,gastrointestinal bleeding
C1391732,tumor cachexia
C0023052,laryngeal edema
C0027429,nasal obstruction
C0024312,lymphocytopenia
C0159060,intestinal noise abnormal
C0221512,stomach pain
C0016382,facial redness
C0600142,hot flash
C0553980,endocardial fibrosis
C0040961,tricuspid regurgitation
C0016382,feeling hot
C0018991,hemiplegia
C0079581,hypochlorhydria
C0240773,plantar erythema
C0085666,spider naevi
C0151514,skin atrophy
C0013312,dupuytren contracture
C0020678,hypotrichosis
C0542571,facial edem
C0018965,erythrocyturia
C1963913,erythrocyte casts present in the sediment
C0151701,pulmonary bleeding
 408 changes: 408 additions & 0 deletions408  
Medical_dataset/SymptomUMLS.csv
Large diffs are not rendered by default.

 42 changes: 42 additions & 0 deletions42  
Medical_dataset/Testing.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,42 @@
itching,skin_rash,nodal_skin_eruptions,continuous_sneezing,shivering,chills,joint_pain,stomach_pain,acidity,ulcers_on_tongue,muscle_wasting,vomiting,burning_micturition,spotting_ urination,fatigue,weight_gain,anxiety,cold_hands_and_feets,mood_swings,weight_loss,restlessness,lethargy,patches_in_throat,irregular_sugar_level,cough,high_fever,sunken_eyes,breathlessness,sweating,dehydration,indigestion,headache,yellowish_skin,dark_urine,nausea,loss_of_appetite,pain_behind_the_eyes,back_pain,constipation,abdominal_pain,diarrhoea,mild_fever,yellow_urine,yellowing_of_eyes,acute_liver_failure,fluid_overload,swelling_of_stomach,swelled_lymph_nodes,malaise,blurred_and_distorted_vision,phlegm,throat_irritation,redness_of_eyes,sinus_pressure,runny_nose,congestion,chest_pain,weakness_in_limbs,fast_heart_rate,pain_during_bowel_movements,pain_in_anal_region,bloody_stool,irritation_in_anus,neck_pain,dizziness,cramps,bruising,obesity,swollen_legs,swollen_blood_vessels,puffy_face_and_eyes,enlarged_thyroid,brittle_nails,swollen_extremeties,excessive_hunger,extra_marital_contacts,drying_and_tingling_lips,slurred_speech,knee_pain,hip_joint_pain,muscle_weakness,stiff_neck,swelling_joints,movement_stiffness,spinning_movements,loss_of_balance,unsteadiness,weakness_of_one_body_side,loss_of_smell,bladder_discomfort,foul_smell_of urine,continuous_feel_of_urine,passage_of_gases,internal_itching,toxic_look_(typhos),depression,irritability,muscle_pain,altered_sensorium,red_spots_over_body,belly_pain,abnormal_menstruation,dischromic _patches,watering_from_eyes,increased_appetite,polyuria,family_history,mucoid_sputum,rusty_sputum,lack_of_concentration,visual_disturbances,receiving_blood_transfusion,receiving_unsterile_injections,coma,stomach_bleeding,distention_of_abdomen,history_of_alcohol_consumption,fluid_overload,blood_in_sputum,prominent_veins_on_calf,palpitations,painful_walking,pus_filled_pimples,blackheads,scurring,skin_peeling,silver_like_dusting,small_dents_in_nails,inflammatory_nails,blister,red_sore_around_nose,yellow_crust_ooze,prognosis
1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Fungal infection
0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Allergy
0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,GERD
1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Chronic cholestasis
1,1,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Drug Reaction
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Peptic ulcer diseae
0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,AIDS
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Diabetes 
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Gastroenteritis
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Bronchial Asthma
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hypertension 
0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Migraine
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Cervical spondylosis
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Paralysis (brain hemorrhage)
1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Jaundice
0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Malaria
1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Chicken pox
0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Dengue
0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Typhoid
0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,hepatitis A
1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hepatitis B
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hepatitis C
0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hepatitis D
0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hepatitis E
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Alcoholic hepatitis
0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,Tuberculosis
0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Common Cold
0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Pneumonia
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Dimorphic hemmorhoids(piles)
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Heart attack
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,Varicose veins
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hypothyroidism
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Hyperthyroidism
0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,Hypoglycemia
0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,Osteoarthristis
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,Arthritis
0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,(vertigo) Paroymsal  Positional Vertigo
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,Acne
0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,Urinary tract infection
0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,Psoriasis
0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,Impetigo
 4,921 changes: 4,921 additions & 0 deletions4,921  
Medical_dataset/Training.csv
Large diffs are not rendered by default.

 2,129 changes: 2,129 additions & 0 deletions2,129  
Medical_dataset/UMLS.ipynb
Large diffs are not rendered by default.

 1,868 changes: 1,868 additions & 0 deletions1,868  
Medical_dataset/dataset_uncleaned.csv
Large diffs are not rendered by default.

 1,183 changes: 1,183 additions & 0 deletions1,183  
Medical_dataset/df_diseases.csv
Large diffs are not rendered by default.

 2,580 changes: 2,580 additions & 0 deletions2,580  
Medical_dataset/disease-symptom-db.csv
Large diffs are not rendered by default.

 1,868 changes: 1,868 additions & 0 deletions1,868  
Medical_dataset/disease-symptoms.csv
Large diffs are not rendered by default.

 1,184 changes: 1,184 additions & 0 deletions1,184  
Medical_dataset/disease_components.csv
Large diffs are not rendered by default.

 10 changes: 10 additions & 0 deletions10  
Medical_dataset/generateConcept.py
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,10 @@
import pandas as pd
import sys

df=pd.read_csv('DiseaseUMLS.csv')
def generate_concept(cui):
    return df.UMLS[df.CUI==cui].values[0]


if __name__=='__main__':
    print(generate_concept(sys.argv[1]))
 1,119 changes: 1,119 additions & 0 deletions1,119  
Medical_dataset/intents.json
Large diffs are not rendered by default.

 1,181 changes: 1,181 additions & 0 deletions1,181  
Medical_dataset/intents_short.json
Large diffs are not rendered by default.

 41 changes: 41 additions & 0 deletions41  
Medical_dataset/symptom_Description.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,41 @@
Drug Reaction,An adverse drug reaction (ADR) is an injury caused by taking medication. ADRs may occur following a single dose or prolonged administration of a drug or result from the combination of two or more drugs.
Malaria,An infectious disease caused by protozoan parasites from the Plasmodium family that can be transmitted by the bite of the Anopheles mosquito or by a contaminated needle or transfusion. Falciparum malaria is the most deadly type.
Allergy,"An allergy is an immune system response to a foreign substance that's not typically harmful to your body.They can include certain foods, pollen, or pet dander. Your immune system's job is to keep you healthy by fighting harmful pathogens."
Hypothyroidism,"Hypothyroidism, also called underactive thyroid or low thyroid, is a disorder of the endocrine system in which the thyroid gland does not produce enough thyroid hormone."
Psoriasis,"Psoriasis is a common skin disorder that forms thick, red, bumpy patches covered with silvery scales. They can pop up anywhere, but most appear on the scalp, elbows, knees, and lower back. Psoriasis can't be passed from person to person. It does sometimes happen in members of the same family."
GERD,"Gastroesophageal reflux disease, or GERD, is a digestive disorder that affects the lower esophageal sphincter (LES), the ring of muscle between the esophagus and stomach. Many people, including pregnant women, suffer from heartburn or acid indigestion caused by GERD."
Chronic cholestasis,"Chronic cholestatic diseases, whether occurring in infancy, childhood or adulthood, are characterized by defective bile acid transport from the liver to the intestine, which is caused by primary damage to the biliary epithelium in most cases"
hepatitis A,Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. The virus is one of several types of hepatitis viruses that cause inflammation and affect your liver's ability to function.
Osteoarthristis,"Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. It occurs when the protective cartilage that cushions the ends of your bones wears down over time."
(vertigo) Paroymsal  Positional Vertigo,Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo — the sudden sensation that you're spinning or that the inside of your head is spinning. Benign paroxysmal positional vertigo causes brief episodes of mild to intense dizziness.
Hypoglycemia, Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. Glucose is your body's main energy source. Hypoglycemia is often related to diabetes treatment. But other drugs and a variety of conditions — many rare — can cause low blood sugar in people who don't have diabetes.
Acne,"Acne vulgaris is the formation of comedones, papules, pustules, nodules, and/or cysts as a result of obstruction and inflammation of pilosebaceous units (hair follicles and their accompanying sebaceous gland). Acne develops on the face and upper trunk. It most often affects adolescents."
Diabetes,"Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy."
Impetigo,"Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects infants and children. Impetigo usually appears as red sores on the face, especially around a child's nose and mouth, and on hands and feet. The sores burst and develop honey-colored crusts."
Hypertension,"Hypertension (HTN or HT), also known as high blood pressure (HBP), is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms."
Peptic ulcer diseae,"Peptic ulcer disease (PUD) is a break in the inner lining of the stomach, the first part of the small intestine, or sometimes the lower esophagus. An ulcer in the stomach is called a gastric ulcer, while one in the first part of the intestines is a duodenal ulcer."
Dimorphic hemorrhoids(piles),"Hemorrhoids, also spelled haemorrhoids, are vascular structures in the anal canal. In their ... Other names, Haemorrhoids, piles, hemorrhoidal disease ."
Common Cold,"The common cold is a viral infection of your nose and throat (upper respiratory tract). It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold."
Chicken pox,"Chickenpox is a highly contagious disease caused by the varicella-zoster virus (VZV). It can cause an itchy, blister-like rash. The rash first appears on the chest, back, and face, and then spreads over the entire body, causing between 250 and 500 itchy blisters."
Cervical spondylosis,"Cervical spondylosis is a general term for age-related wear and tear affecting the spinal disks in your neck. As the disks dehydrate and shrink, signs of osteoarthritis develop, including bony projections along the edges of bones (bone spurs)."
Hyperthyroidism,"Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the hormone thyroxine. Hyperthyroidism can accelerate your body's metabolism, causing unintentional weight loss and a rapid or irregular heartbeat."
Urinary tract infection,"Urinary tract infection: An infection of the kidney, ureter, bladder, or urethra. Abbreviated UTI. Not everyone with a UTI has symptoms, but common symptoms include a frequent urge to urinate and pain or burning when urinating."
Varicose veins,"A vein that has enlarged and twisted, often appearing as a bulging, blue blood vessel that is clearly visible through the skin. Varicose veins are most common in older adults, particularly women, and occur especially on the legs."
AIDS,"Acquired immunodeficiency syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). By damaging your immune system, HIV interferes with your body's ability to fight infection and disease."
Paralysis (brain hemorrhage),"Intracerebral hemorrhage (ICH) is when blood suddenly bursts into brain tissue, causing damage to your brain. Symptoms usually appear suddenly during ICH. They include headache, weakness, confusion, and paralysis, particularly on one side of your body."
Typhoid,"An acute illness characterized by fever caused by infection with the bacterium Salmonella typhi. Typhoid fever has an insidious onset, with fever, headache, constipation, malaise, chills, and muscle pain. Diarrhea is uncommon, and vomiting is not usually severe."
Hepatitis B,"Hepatitis B is an infection of your liver. It can cause scarring of the organ, liver failure, and cancer. It can be fatal if it isn't treated. It's spread when people come in contact with the blood, open sores, or body fluids of someone who has the hepatitis B virus."
Fungal infection,"In humans, fungal infections occur when an invading fungus takes over an area of the body and is too much for the immune system to handle. Fungi can live in the air, soil, water, and plants. There are also some fungi that live naturally in the human body. Like many microbes, there are helpful fungi and harmful fungi."
Hepatitis C,"Inflammation of the liver due to the hepatitis C virus (HCV), which is usually spread via blood transfusion (rare), hemodialysis, and needle sticks. The damage hepatitis C does to the liver can lead to cirrhosis and its complications as well as cancer."
Migraine,"A migraine can cause severe throbbing pain or a pulsing sensation, usually on one side of the head. It's often accompanied by nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days, and the pain can be so severe that it interferes with your daily activities."
Bronchial Asthma,"Bronchial asthma is a medical condition which causes the airway path of the lungs to swell and narrow. Due to this swelling, the air path produces excess mucus making it hard to breathe, which results in coughing, short breath, and wheezing. The disease is chronic and interferes with daily working."
Alcoholic hepatitis,"Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period of time. It's also aggravated by binge drinking and ongoing alcohol use. If you develop this condition, you must stop drinking alcohol"
Jaundice,"Yellow staining of the skin and sclerae (the whites of the eyes) by abnormally high blood levels of the bile pigment bilirubin. The yellowing extends to other tissues and body fluids. Jaundice was once called the ""morbus regius"" (the regal disease) in the belief that only the touch of a king could cure it"
Hepatitis E,A rare form of liver inflammation caused by infection with the hepatitis E virus (HEV). It is transmitted via food or drink handled by an infected person or through infected water supplies in areas where fecal matter may get into the water. Hepatitis E does not cause chronic liver disease.
Dengue,"an acute infectious disease caused by a flavivirus (species Dengue virus of the genus Flavivirus), transmitted by aedes mosquitoes, and characterized by headache, severe joint pain, and a rash. — called also breakbone fever, dengue fever."
Hepatitis D,"Hepatitis D, also known as the hepatitis delta virus, is an infection that causes the liver to become inflamed. This swelling can impair liver function and cause long-term liver problems, including liver scarring and cancer. The condition is caused by the hepatitis D virus (HDV)."
Heart attack,"The death of heart muscle due to the loss of blood supply. The loss of blood supply is usually caused by a complete blockage of a coronary artery, one of the arteries that supplies blood to the heart muscle."
Pneumonia,"Pneumonia is an infection in one or both lungs. Bacteria, viruses, and fungi cause it. The infection causes inflammation in the air sacs in your lungs, which are called alveoli. The alveoli fill with fluid or pus, making it difficult to breathe."
Arthritis,"Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of arthritis are joint pain and stiffness, which typically worsen with age. The most common types of arthritis are osteoarthritis and rheumatoid arthritis."
Gastroenteritis,"Gastroenteritis is an inflammation of the digestive tract, particularly the stomach, and large and small intestines. Viral and bacterial gastroenteritis are intestinal infections associated with symptoms of diarrhea , abdominal cramps, nausea , and vomiting ."
Tuberculosis,"Tuberculosis (TB) is an infectious disease usually caused by Mycobacterium tuberculosis (MTB) bacteria. Tuberculosis generally affects the lungs, but can also affect other parts of the body. Most infections show no symptoms, in which case it is known as latent tuberculosis."
 41 changes: 41 additions & 0 deletions41  
Medical_dataset/symptom_precaution.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,41 @@
Drug Reaction,stop irritation,consult nearest hospital,stop taking drug,follow up
Malaria,Consult nearest hospital,avoid oily food,avoid non veg food,keep mosquitos out
Allergy,apply calamine,cover area with bandage,,use ice to compress itching
Hypothyroidism,reduce stress,exercise,eat healthy,get proper sleep
Psoriasis,wash hands with warm soapy water,stop bleeding using pressure,consult doctor,salt baths
GERD,avoid fatty spicy food,avoid lying down after eating,maintain healthy weight,exercise
Chronic cholestasis,cold baths,anti itch medicine,consult doctor,eat healthy
hepatitis A,Consult nearest hospital,wash hands through,avoid fatty spicy food,medication
Osteoarthristis,acetaminophen,consult nearest hospital,follow up,salt baths
(vertigo) Paroymsal  Positional Vertigo,lie down,avoid sudden change in body,avoid abrupt head movment,relax
Hypoglycemia,lie down on side,check in pulse,drink sugary drinks,consult doctor
Acne,bath twice,avoid fatty spicy food,drink plenty of water,avoid too many products
Diabetes ,have balanced diet,exercise,consult doctor,follow up
Impetigo,soak affected area in warm water,use antibiotics,remove scabs with wet compressed cloth,consult doctor
Hypertension ,meditation,salt baths,reduce stress,get proper sleep
Peptic ulcer diseae,avoid fatty spicy food,consume probiotic food,eliminate milk,limit alcohol
Dimorphic hemmorhoids(piles),avoid fatty spicy food,consume witch hazel,warm bath with epsom salt,consume alovera juice
Common Cold,drink vitamin c rich drinks,take vapour,avoid cold food,keep fever in check
Chicken pox,use neem in bathing ,consume neem leaves,take vaccine,avoid public places
Cervical spondylosis,use heating pad or cold pack,exercise,take otc pain reliver,consult doctor
Hyperthyroidism,eat healthy,massage,use lemon balm,take radioactive iodine treatment
Urinary tract infection,drink plenty of water,increase vitamin c intake,drink cranberry juice,take probiotics
Varicose veins,lie down flat and raise the leg high,use oinments,use vein compression,dont stand still for long
AIDS,avoid open cuts,wear ppe if possible,consult doctor,follow up
Paralysis (brain hemorrhage),massage,eat healthy,exercise,consult doctor
Typhoid,eat high calorie vegitables,antiboitic therapy,consult doctor,medication
Hepatitis B,consult nearest hospital,vaccination,eat healthy,medication
Fungal infection,bath twice,use detol or neem in bathing water,keep infected area dry,use clean cloths
Hepatitis C,Consult nearest hospital,vaccination,eat healthy,medication
Migraine,meditation,reduce stress,use poloroid glasses in sun,consult doctor
Bronchial Asthma,switch to loose cloothing,take deep breaths,get away from trigger,seek help
Alcoholic hepatitis,stop alcohol consumption,consult doctor,medication,follow up
Jaundice,drink plenty of water,consume milk thistle,eat fruits and high fiberous food,medication
Hepatitis E,stop alcohol consumption,rest,consult doctor,medication
Dengue,drink papaya leaf juice,avoid fatty spicy food,keep mosquitos away,keep hydrated
Hepatitis D,consult doctor,medication,eat healthy,follow up
Heart attack,call ambulance,chew or swallow asprin,keep calm,
Pneumonia,consult doctor,medication,rest,follow up
Arthritis,exercise,use hot and cold therapy,try acupuncture,massage
Gastroenteritis,stop eating solid food for while,try taking small sips of water,rest,ease back into eating
Tuberculosis,cover mouth,consult doctor,medication,rest
 133 changes: 133 additions & 0 deletions133  
Medical_dataset/symptom_severity.csv
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,133 @@
itching,1
skin_rash,3
nodal_skin_eruptions,4
continuous_sneezing,4
shivering,5
chills,3
joint_pain,3
stomach_pain,5
acidity,3
ulcers_on_tongue,4
muscle_wasting,3
vomiting,5
burning_micturition,6
spotting_urination,6
fatigue,4
weight_gain,3
anxiety,4
cold_hands_and_feets,5
mood_swings,3
weight_loss,3
restlessness,5
lethargy,2
patches_in_throat,6
irregular_sugar_level,5
cough,4
high_fever,7
sunken_eyes,3
breathlessness,4
sweating,3
dehydration,4
indigestion,5
headache,3
yellowish_skin,3
dark_urine,4
nausea,5
loss_of_appetite,4
pain_behind_the_eyes,4
back_pain,3
constipation,4
abdominal_pain,4
diarrhoea,6
mild_fever,5
yellow_urine,4
yellowing_of_eyes,4
acute_liver_failure,6
fluid_overload,6
swelling_of_stomach,7
swelled_lymph_nodes,6
malaise,6
blurred_and_distorted_vision,5
phlegm,5
throat_irritation,4
redness_of_eyes,5
sinus_pressure,4
runny_nose,5
congestion,5
chest_pain,7
weakness_in_limbs,7
fast_heart_rate,5
pain_during_bowel_movements,5
pain_in_anal_region,6
bloody_stool,5
irritation_in_anus,6
neck_pain,5
dizziness,4
cramps,4
bruising,4
obesity,4
swollen_legs,5
swollen_blood_vessels,5
puffy_face_and_eyes,5
enlarged_thyroid,6
brittle_nails,5
swollen_extremeties,5
excessive_hunger,4
extra_marital_contacts,5
drying_and_tingling_lips,4
slurred_speech,4
knee_pain,3
hip_joint_pain,2
muscle_weakness,2
stiff_neck,4
swelling_joints,5
movement_stiffness,5
spinning_movements,6
loss_of_balance,4
unsteadiness,4
weakness_of_one_body_side,4
loss_of_smell,3
bladder_discomfort,4
foul_smell_ofurine,5
continuous_feel_of_urine,6
passage_of_gases,5
internal_itching,4
toxic_look_(typhos),5
depression,3
irritability,2
muscle_pain,2
altered_sensorium,2
red_spots_over_body,3
belly_pain,4
abnormal_menstruation,6
dischromic_patches,6
watering_from_eyes,4
increased_appetite,5
polyuria,4
family_history,5
mucoid_sputum,4
rusty_sputum,4
lack_of_concentration,3
visual_disturbances,3
receiving_blood_transfusion,5
receiving_unsterile_injections,2
coma,7
stomach_bleeding,6
distention_of_abdomen,4
history_of_alcohol_consumption,5
fluid_overload,4
blood_in_sputum,5
prominent_veins_on_calf,6
palpitations,4
painful_walking,2
pus_filled_pimples,2
blackheads,2
scurring,2
skin_peeling,3
silver_like_dusting,2
small_dents_in_nails,2
inflammatory_nails,2
blister,4
red_sore_around_nose,2
yellow_crust_ooze,3

 516 changes: 516 additions & 0 deletions516  
Medical_dataset/tfidfsymptoms.csv
Large diffs are not rendered by default.

 275 changes: 275 additions & 0 deletions275  
Medical_dataset/translate GER2ENG Diseases.ipynb
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,275 @@
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Detected(lang=de, confidence=0.90267855)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "from googletrans import Translator\n",
    "\n",
    "detector = Translator()\n",
    "\n",
    "dec_lan = detector.detect('der hund')\n",
    "\n",
    "print(dec_lan)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "gd=pd.read_csv('disease-symptom-db.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "disease=gd[['Disease_UMLS','Disease_CUI']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Disease_UMLS</th>\n",
       "      <th>Disease_CUI</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>porphyrie , akute intermittierende</td>\n",
       "      <td>C0162565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>porphyria cutanea tarda</td>\n",
       "      <td>C0162566</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>vorhofflimmern</td>\n",
       "      <td>C0004238</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>linksherzinsuffizienz</td>\n",
       "      <td>C0023212</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>rechtsherzinsuffizienz</td>\n",
       "      <td>C0235527</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2543</th>\n",
       "      <td>leberversagen , akutes</td>\n",
       "      <td>C0162557</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2551</th>\n",
       "      <td>postinfektioese glomerulonephritis</td>\n",
       "      <td>C3805004</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2565</th>\n",
       "      <td>rapid - progressives nephritisches syndrom</td>\n",
       "      <td>C0451728</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2571</th>\n",
       "      <td>renales osteopathiesyndrom</td>\n",
       "      <td>C3805154</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2573</th>\n",
       "      <td>nephroblastom</td>\n",
       "      <td>C0027708</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>312 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                    Disease_UMLS Disease_CUI\n",
       "0             porphyrie , akute intermittierende    C0162565\n",
       "6                        porphyria cutanea tarda    C0162566\n",
       "8                                 vorhofflimmern    C0004238\n",
       "20                         linksherzinsuffizienz    C0023212\n",
       "35                        rechtsherzinsuffizienz    C0235527\n",
       "...                                          ...         ...\n",
       "2543                      leberversagen , akutes    C0162557\n",
       "2551          postinfektioese glomerulonephritis    C3805004\n",
       "2565  rapid - progressives nephritisches syndrom    C0451728\n",
       "2571                  renales osteopathiesyndrom    C3805154\n",
       "2573                               nephroblastom    C0027708\n",
       "\n",
       "[312 rows x 2 columns]"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "disease.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [],
   "source": [
    "disease.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "disease=disease.loc[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "disease['Disease'] = disease.Disease_UMLS.apply(detector.translate, src='de', dest='en').apply(getattr, args=('text',))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'vorhofflimmern'"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "detector.translate('vorhofflimmern', src='de', dest='en').text"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "symp=gd[['Symptom','Symptom_CUI']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "symp.drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "symp.reset_index(drop=True,inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
 1 change: 1 addition & 0 deletions1  
Procfile
Original file line number	Diff line number	Diff line change
@@ -0,0 +1 @@
web: gunicorn app:app
 22 changes: 22 additions & 0 deletions22  
README.md
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,22 @@
# ROBO-DOC
Healthcare chatbot to predict Diseases based on patient symptoms.
<br>
<p align="center">
  <img src="screens\prediction.png" width="500" >
</p>

# How to use:
## create a venv 
virtualenv venv 

## activate it and install reqs
source venv/bin/activate
pip install -r requirements.txt 
python -m spacy download en_core_web_sm

## run app file
python app.py


Medical DataSet available !!
---- 
 2,645 changes: 2,645 additions & 0 deletions2,645  
ROBO_DOC .ipynb
Large diffs are not rendered by default.

 656 changes: 656 additions & 0 deletions656  
app.py
Large diffs are not rendered by default.

 Binary file addedBIN +4.99 MB 
model/knn.pkl
Binary file not shown.
 516 changes: 516 additions & 0 deletions516  
model/tfidfsymptoms.csv
Large diffs are not rendered by default.

 7 changes: 7 additions & 0 deletions7  
requirements.txt
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,7 @@
numpy
nltk
spacy
joblib
pandas
Flask
scikit_learn
 Binary file addedBIN +156 KB 
screens/description.png
Loading
 Binary file addedBIN +147 KB 
screens/precautions.png
Loading
 Binary file addedBIN +176 KB 
screens/prediction.png
Loading
 Binary file addedBIN +162 KB 
screens/sym1.png
Loading
 Binary file addedBIN +157 KB 
screens/sym2.png
Loading
 152 changes: 152 additions & 0 deletions152  
static/styles/style.css
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,152 @@
:root {
    --body-bg: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    --msger-bg: #fff;
    --border: 2px solid #ddd;
    --left-msg-bg: #ececec;
    --right-msg-bg: #579ffb;
  }

  html {
    box-sizing: border-box;
  }

  *,
  *:before,
  *:after {
    margin: 0;
    padding: 0;
    box-sizing: inherit;
  }

  body {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    background-image: var(--body-bg);
    font-family: Helvetica, sans-serif;
  }

  .msger {
    display: flex;
    flex-flow: column wrap;
    justify-content: space-between;
    width: 100%;
    max-width: 867px;
    margin: 25px 10px;
    height: calc(100% - 50px);
    border: var(--border);
    border-radius: 5px;
    background: var(--msger-bg);
    box-shadow: 0 15px 15px -5px rgba(0, 0, 0, 0.2);
  }

  .msger-header {
    /* display: flex; */
    font-size: medium;
    justify-content: space-between;
    padding: 10px;
    text-align: center;
    border-bottom: var(--border);
    background: #eee;
    color: #666;
  }

  .msger-chat {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
  }
  .msger-chat::-webkit-scrollbar {
    width: 6px;
  }
  .msger-chat::-webkit-scrollbar-track {
    background: #ddd;
  }
  .msger-chat::-webkit-scrollbar-thumb {
    background: #bdbdbd;
  }
  .msg {
    display: flex;
    align-items: flex-end;
    margin-bottom: 10px;
  }

  .msg-img {
    width: 50px;
    height: 50px;
    margin-right: 10px;
    background: #ddd;
    background-repeat: no-repeat;
    background-position: center;
    background-size: cover;
    border-radius: 50%;
  }
  .msg-bubble {
    max-width: 450px;
    padding: 15px;
    border-radius: 15px;
    background: var(--left-msg-bg);
  }
  .msg-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .msg-info-name {
    margin-right: 10px;
    font-weight: bold;
  }
  .msg-info-time {
    font-size: 0.85em;
  }

  .left-msg .msg-bubble {
    border-bottom-left-radius: 0;
  }

  .right-msg {
    flex-direction: row-reverse;
  }
  .right-msg .msg-bubble {
    background: var(--right-msg-bg);
    color: #fff;
    border-bottom-right-radius: 0;
  }
  .right-msg .msg-img {
    margin: 0 0 0 10px;
  }

  .msger-inputarea {
    display: flex;
    padding: 10px;
    border-top: var(--border);
    background: #eee;
  }
  .msger-inputarea * {
    padding: 10px;
    border: none;
    border-radius: 3px;
    font-size: 1em;
  }
  .msger-input {
    flex: 1;
    background: #ddd;
  }
  .msger-send-btn {
    margin-left: 10px;
    background: rgb(0, 196, 65);
    color: #fff;
    font-weight: bold;
    cursor: pointer;
    transition: background 0.23s;
  }
  .msger-send-btn:hover {
    background: rgb(0, 180, 50);
  }

  .msger-chat {
    background-color: #fcfcfe;
    background-image: url("test.jpg")
  }
 Binary file addedBIN +45.4 KB 
static/styles/test.jpg
Loading
 Binary file addedBIN +49.2 KB 
static/styles/test2.jpg
Loading
 Binary file addedBIN +12.5 KB 
static/styles/testt.jpg
Loading
 134 changes: 134 additions & 0 deletions134  
templates/home.html
Original file line number	Diff line number	Diff line change
@@ -0,0 +1,134 @@
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <title>Robo-Doc</title>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="X-UA-Compatible" content="ie=edge">
  <link rel="stylesheet" href="{{ url_for('static', filename='styles/style.css') }}">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
</head>

<body>
  <!-- partial:index.partial.html -->
  <section class="msger">
    <header class="msger-header">
      <div class="msger-header-title">
        ROBO-DOC
      </div>
    </header>

    <main class="msger-chat">
      <div class="msg left-msg">
        <div class="msg-img" style="background-image: url(https://image.flaticon.com/icons/svg/327/327779.svg)"></div>
        <div class="msg-bubble">
          <div class="msg-info">
            <div class="msg-info-name">Robo-Doc</div>
          </div>
          <div class="msg-text">
             Hello, my name is RoboDoc, and I will be happy to help diagnose your disease. 
          </div>
        </div>
      </div>
      <div class="msg left-msg">
        <div class="msg-img" style="background-image: url(https://image.flaticon.com/icons/svg/327/327779.svg)"></div>

        <div class="msg-bubble">
          <div class="msg-info">
            <div class="msg-info-name">Robo-Doc</div>
          </div>
          <div class="msg-text">
            To start, we need to ask some basic questions, tap OK to continue ! 
          </div>
        </div>
      </div>
    </main>


    <form class="msger-inputarea">
      <input type="text" class="msger-input" id="textInput" placeholder="Enter your message...">
      <button type="submit" class="msger-send-btn">Send</button>
    </form>
  </section>
  <!-- partial -->
  <script src='https://use.fontawesome.com/releases/v5.0.13/js/all.js'></script>
  <script>

    const msgerForm = get(".msger-inputarea");
    const msgerInput = get(".msger-input");
    const msgerChat = get(".msger-chat");


    // Icons made by Freepik from www.flaticon.com
    const BOT_IMG = "https://image.flaticon.com/icons/svg/327/327779.svg";
    const PERSON_IMG = "https://image.flaticon.com/icons/svg/145/145867.svg";
    const BOT_NAME = "Robo-Doc";
    const PERSON_NAME = "You";

    msgerForm.addEventListener("submit", event => {
      event.preventDefault();

      const msgText = msgerInput.value;
      if (!msgText) return;

      appendMessage(PERSON_NAME, PERSON_IMG, "right", msgText);
      msgerInput.value = "";
      botResponse(msgText);
    });

    function appendMessage(name, img, side, text) {
      //   Simple solution for small apps
      const msgHTML = `
<div class="msg ${side}-msg">
  <div class="msg-img" style="background-image: url(${img})"></div>
  <div class="msg-bubble">
    <div class="msg-info">
      <div class="msg-info-name">${name}</div>
      <div class="msg-info-time">${formatDate(new Date())}</div>
    </div>
    <div class="msg-text">${text}</div>
  </div>
</div>
`;

      msgerChat.insertAdjacentHTML("beforeend", msgHTML);
      msgerChat.scrollTop += 500;
    }

    function botResponse(rawText) {

      // Bot Response
      $.get("/get", { msg: rawText }).done(function (data) {
        console.log(rawText);
        console.log(data);
        const msgText = data;
        appendMessage(BOT_NAME, BOT_IMG, "left", msgText);

      });

    }


    // Utils
    function get(selector, root = document) {
      return root.querySelector(selector);
    }

    function formatDate(date) {
      const h = "0" + date.getHours();
      const m = "0" + date.getMinutes();

      return `${h.slice(-2)}:${m.slice(-2)}`;
    }



  </script>

</body>

</html>
