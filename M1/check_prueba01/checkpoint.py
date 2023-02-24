# Importante: No modificar ni el nombre ni los argumetos que reciben las funciones, sólo deben escribir
# código dentro de las funciones ya definidas.

# Recordar utilizar la ruta relativa, no la absoluta para ingestar los datos desde los CSV
# EJ: 'datasets/xxxxxxxxxx.csv'
import pandas as pd
import numpy as np

def Ret_Pregunta01():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file.
    "Fuentes_Consumo_Energia.csv".
    This function must report the number of records whose entity is Colombia or Mexico returning that value in a data
    tuple (Colombia, Mexico).
    Hint to find out the function Shape
    '''
     #Your code here:
    '''
    This code reads the CSV file into a pandas dataframe, 
    filters the dataframe to only include records where the entity is either Colombia or Mexico,
    and then uses the shape attribute of the filtered dataframe
    to count the number of records for each country. 
    Finally, it returns the counts as a tuple.
    '''

    # Load the CSV file into a pandas dataframe
    df = pd.read_csv("datasets/Fuentes_Consumo_Energia.csv",
                     sep=',')

    # Filter the dataframe to only include records where entity is Colombia or Mexico
    df_filtered = df[(df["Entity"] == "Colombia") | (df["Entity"] == "Mexico")]

    # Get the number of records for each country using the shape attribute of the filtered dataframe
    colombia_count = df_filtered[df_filtered["Entity"] == "Colombia"].shape[0]
    mexico_count = df_filtered[df_filtered["Entity"] == "Mexico"].shape[0]

    # Return the counts as a tuple
    return (colombia_count, mexico_count)

def Ret_Pregunta02():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file.
    "Fuentes_Consumo_Energia.csv".
    This function must remove the columns 'Code' and 'Entity' and then report the number of columns by returning that value in an integer data type.
    returning that value in an integer data type.
    '''
    #Your code here:
    '''
    This code reads in the CSV file using pd.read_csv(), 
    drops the 'Code' and 'Entity' columns using df.drop(), 
    gets the number of remaining columns using len(df.columns), 
    and returns the result as an integer.
    '''
    df = pd.read_csv('datasets/Fuentes_Consumo_Energia.csv')
    df = df.drop(columns=['Code', 'Entity'])
    num_cols = len(df.columns)
    return num_cols


def Ret_Pregunta03():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file.
    "Fuentes_Consumo_Energia.csv".
    This function should report the number of records in the Year column without taking into account 
    those with missing values returning that value in an integer data type.
    '''
    #Your code here:
    '''
    This function first reads the CSV file into a DataFrame object using the read_csv function
    from Pandas.
    Then it uses the count method of the Year column to count the number of non-missing values
    in that column. (Count doesn't consider Null/None values)
    Finally, it returns the count as an integer.
    '''
    # Read the CSV file into a DataFrame
    df = pd.read_csv("datasets/Fuentes_Consumo_Energia.csv")

    # Count the number of non-missing values in the Year column
    num_records = df['Year'].count()

    return num_records

def Ret_Pregunta04():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file.
    "Fuentes_Consumo_Energia.csv".
    The "ExaJoule" is a different unit than the TWh, that is, it does not make sense to add them or to look for proportions between them.
    the conversion formula is:
    277,778 Terawatt/Hour (TWh) = 1 Exajoule.
    The fields ending in "_EJ" correspond to measurements in Exajoules,
    and those ending in "_TWh" correspond to Terawatt/Hour.
    The instruction is to create a new field, called "Total_Consumption", which stores the sum of the measurements in Exajoules.
    and that stores the sum of all the consumptions expressed in Terawatt/Hour (converting to this measure those that are expressed in Terawatt/Hour).
    (converting to this measure those that are in Exajoules).
    This function should report the total consumption for the entity 'World' and year '2019',
    rounded to 2 decimal places, returning this value in a float data.
    '''
    #Your code
    df = pd.read_csv("datasets/Fuentes_Consumo_Energia.csv")
    #Get EJ columns
    EJ_columns = [i for i in df.columns if '_EJ' in i]
    #Get TWh columns
    TWH_columns = [i for i in df.columns if '_TWh' in i]
    #Add consuptions
    df['Consumo_Total'] = df[TWH_columns].sum(axis=1) + (df[EJ_columns]*277.778).sum(axis = 1)
    #filter
    df = df[(df["Entity"] == "World") & (df["Year"] == 2019)]

    return round(float(df['Consumo_Total']), 2)

def Ret_Pregunta05():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file.
    "Fuentes_Consumo_Energia.csv".
    This function must report the year of the highest hydro generation (Hydro_Generation_TWh)
    for the entity 'Europe' returning that value in an integer data type.
    '''
    # your code:
    df = pd.read_csv("/datasets/Fuentes_Consumo_Energia.csv")
    df = df[df["Entity"] == "Europe"]
    index = df["Hydro_Generation_TWh"].idxmax()
    

    return df["Year"][index]

def Ret_Pregunta06(m1, m2, m3):
    '''
    This function receives three Numpy arrays of 2 dimensions each, and returns the boolean value.
    True if it is possible to perform a multiplication between the three arrays (n1 x n2 x n3),
    and the boolean value False if it is not.
    E.g:
        n1 = np.array([[0,0,0],[1,1,1],[2,2,2]])
        n2 = np.array([[3,3],[4,4],[5,5]])
        n3 = np.array([1,1],[2,2])
        print(Ret_Question06(n1,n2,n3))
            True -> Value returned by the function in this example
        print(Ret_Question06(n2,n1,n3))
            False -> Value returned by the function in this example
    '''
    #Tu código aca:
    '''
    Explanation:
        To determine if it is possible to perform a multiplication between the three arrays
        (n1 x n2 x n3), we need to verify if the number of columns of the first matrix (n1)
        is equal to the number of rows of the second matrix (n2), and if the number of columns
        of the second matrix (n2) is equal to the number of rows of the third matrix (n3).
    '''

    '''
    This function receives three Numpy arrays of 2 dimensions each,
    and returns the boolean value. 
    It checks if it is possible to perform a multiplication between the three arrays 
    (n1 x n2 x n3) and returns True if it is possible and False if it is not.
    '''

    if m1.shape[1] == m2.shape[0] and m2.shape[1] == m3.shape[0]:
        return True
    else:
        return False
    
def Ret_Pregunta07():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file
    ''Sources_Energy_Consumption.csv''.
    This function should report which of the following list of countries had the highest generation of hydro energy (Hydro_Generation_TW).
    (Hydro_Generation_TWh) in the year 2019:
        * Argentina
        * Brazil
        * Chile
        * Colombia
        * Ecuador
        * Mexico
        * Peru
    You must return the value in a string data type.
    '''
    # Your code:
    df = pd.read_csv("datasets/Fuentes_Consumo_Energia.csv")

    countries = ["Argentina","Brazil","Chile","Colombia","Ecuador","Mexico","Peru"]

    mask = (df["Entity"].isin(countries)) & (df["Year"] == 2019)
    df = df[mask]

    index = df.Hydro_Generation_TWh.idxmax()


    return  df["Entity"][index]

def Ret_Pregunta08():
    '''
    You must use Pandas to ingest in a Dataframe object the content of the provided file
    "Sources_Energy_Consumption.csv".
    This function must report the amount of different entities that are present in the dataset
    returning that value in an integer type data.
    '''
    # Your code:
    df = pd.read_csv("datasets/Fuentes_Consumo_Energia.csv")

    return df["Entity"].nunique()

def Ret_Pregunta09():
   '''
    You must use Pandas to ingest into a Dataframe object the contents of the provided file.
    "datasets/Table1_exercise_table.csv" and "datasets/Table2_exercise.csv".
    This function must report: score_average_female and score_average_male in tuple format.   
    
   '''
   # Your code:
    
   df_Table1 = pd.read_csv("./datasets/Tabla1_ejercicio.csv", sep=';')
   df_Table2 = pd.read_csv('./datasets/Tabla2_ejercicio.csv', sep=';')
   df_final = pd.merge(df_Table1,df_Table2,on="pers_id").drop_duplicates()
    
   score_average_female = round(df_final[df_final.sexo=="F"].score.mean(),2)
   score_average_male = round(df_final[df_final.sexo=="M"].score.mean(),2)
    
   return (score_average_female, score_average_male)

def Ret_Pregunta10(lista):
    '''
    This function receives as parameter an object of the List() class defined in the file List.py.
    It must traverse the list and return the number of nodes it has. Use the method of the class
    List called getCabecera()
    Example:
        lis = List()
        list.addElement(1)
        list.addElement(2)
        list.addElement(3)
        print(Ret_Question10(list))
            3 -> Must be the value returned by the function Ret_Pregungunta10() in this example
    '''
    '''
    This function first gets the head (header) of the linked list using the getCabecera()
    method of the Lista class. 
    If the cabecera is None, that means the list is empty, so the function returns 0.

    If the list is not empty, the function initializes a counter variable to 1 and
    sets a pointer variable to the head.
    The function then enters a loop that iterates over each node in the list 
    until it reaches the end. 
    For each node, it increments the contador by 1 and moves the pointer
    to the next node.
    After the loop finishes, the counter variable contains the total number
    of nodes in the list, so the function returns that value.
    '''

    # Your code here
    header= list.getCabecera()
    if header is None:
        return 0
    counter = 1
    pointer = header
    while pointer.getSiguiente() is not None:
        counter += 1
        puntero = puntero.getSiguiente()
    return counter