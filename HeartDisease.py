#import library 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


import random
import math





#Read data
data=pd.read_csv("F:\EELU\C\S2\AI-2\Project\heart.csv")
print(data)
print("*"*50)
print("*"*50)



#EDA
print("-"*10,"EDA","-"*10)
print("** Dataframe information before preprocessing **\n")
print(f"number of rows: {data.shape[0]}\nnumber of columns: {data.shape[1]}")
print('***'*10)
print(data.describe().T)
print('***'*10)
print(f"null values in each column:\n\n{data.isnull().sum()}\n\nSum of duplicates: {data.duplicated().sum()}")
print("*"*50)
print("*"*50)





#Remove outliers
print("-"*10,"Remove Outliers","-"*10)
col=["RestingBP","Cholesterol","MaxHR"]
for i in col:
    upper_limit = data[i].quantile(0.99)
    lower_limit = data[i].quantile(0.01)
    print(f'upper limit data{i}:', upper_limit)
    print(f'lower limit data{i}:', lower_limit)
    print("*"*10)
    new_data = data.loc[(data[i] <= upper_limit) & (data[i] >= lower_limit)]
print("*"*50)
print("*"*50)






# Encoding
print("-"*10,"Encoding","-"*10)
# Create a LabelEncoder object
le = LabelEncoder()
# Select categorical columns
categorical_cols = data.select_dtypes(include='object').columns

# Encode each categorical column
for col in categorical_cols:
    new_data[col] = le.fit_transform(new_data[col])

# Print the updated DataFrame
new_data.head()

print("*"*50)
print("*"*50)




#Split data
x=new_data.drop(["HeartDisease"],axis=1)
y=new_data["HeartDisease"]

x_train , x_test , y_train ,y_test = train_test_split(x,y , test_size= 0.25,random_state=40 )
print("data shape is:",new_data.shape)
print("Xtrain shape is:",x_train.shape)
print("Xtest shape is:",x_test.shape)




#Logistic Regression

print("-"*10,"Logistic Regression","-"*10)
lr = LogisticRegression()
lr.fit(x_train,y_train)
prediction_lr=lr.predict(x_test)
print("acuuricy score: ", accuracy_score(prediction_lr,y_test))
print(confusion_matrix(prediction_lr,y_test))
print("*"*50)



#Decision Tress
print("-"*10,"Decision trees","-"*10)
dt=DecisionTreeClassifier()
dt.fit(x_train, y_train)
prediction_dt=dt.predict(x_test)
print('The accuracy of the Decision Tree is',accuracy_score(prediction_dt,y_test))
print (classification_report(y_test , prediction_dt))
print("*"*50)


#Random Forest
print("-"*10,"random Forest","-"*10)
rf = RandomForestClassifier(n_estimators= 18 , max_depth= 8 , max_features= 9)
rf.fit(x_train, y_train)
prediction_rf=rf.predict(x_test)
print('The accuracy of the Decision Tree is',accuracy_score(prediction_rf,y_test))
print (classification_report(y_test , prediction_rf))
print("*"*50)



#PCA
print("-"*10,"PCA","-"*10)
pca = PCA(n_components=1)
pca.fit_transform(x)
print(pca.explained_variance_ratio_) #h value
print(pca.singular_values_)
print("*"*50)





import pickle
pickle.dump(dt, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print("check HeartDisease: ",model.predict([[49,	0,	2,	160,	180,	0,	1,	156,	0,	1.0,	1	]]))


print("*"*50)
print("-"*10,"Differential_Evolution_Algorithm","-"*10)

def obj(x):
    return np.sum(x**2)


def mutation(individuals, F):
    a, b, c = individuals
    return a + F * (b - c)


def check_bounds(vec, bounds):
    vec = np.clip(vec, bounds[:, 0], bounds[:, 1])
    return vec


def crossover(mutated, target, cr, dims):
    p = np.random.rand(dims)
    trial = np.where(p < cr, mutated, target)
    return trial

def differential_evolution(pop_size, bounds, iter, F, cr):
    dims = len(bounds)
    pop = bounds[:, 0] + np.random.rand(pop_size, dims) * (bounds[:, 1] - bounds[:, 0])
    obj_all = np.array([obj(ind) for ind in pop])
    best_idx = np.argmin(obj_all)
    best_vector = pop[best_idx]
    best_obj = obj_all[best_idx]

    for i in range(iter):
        for j in range(pop_size):
            candidates = [idx for idx in range(pop_size) if idx != j]
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            mutated = mutation([a, b, c], F)
            mutated = check_bounds(mutated, bounds)
            trial = crossover(mutated, pop[j], cr, dims)
            obj_trial = obj(trial)
            if obj_trial < obj_all[j]:
                pop[j] = trial
                obj_all[j] = obj_trial
                if obj_trial < best_obj:
                    best_vector = trial
                    best_obj = obj_trial
        print(f'Iteration {i + 1}: Best solution f({np.round(best_vector, 5)}) = {best_obj:.5f}')

    return best_vector, best_obj



# Parameters and execution of the algorithm
bounds = np.array([(-5.0, 5.0), (-5.0, 5.0)])
pop_size = 10
iterations = 100
F = 0.5  # Mutation factor
cr = 0.7  # Crossover rate

best_vector, best_value = differential_evolution(pop_size, bounds, iterations, F, cr)
print(f'\nFinal Solution: f({np.round(best_vector, 5)}) = {best_value:.5f}')

# Plotting the results
plt.plot(np.arange(iterations), '.-')
plt.xlabel('Iteration Number')
plt.ylabel('Best Objective Value')
plt.title('Objective Function Value by Iteration')
plt.show()


print("*"*50)
print("-"*10,"Genatic Algorithms","-"*10)



# Define the parameters
TARGET = 0  # The target value we want to reach
POPULATION_SIZE = 4
MUTATION_RATE = 0.01
NUM_GENES = 5  # Number of genes in an individual

# Define the range for the genes
GENE_MIN = -10
GENE_MAX = 10




# Generate initial population
def generate_population(size, num_genes):
    population = []
    for _ in range(size):
        individual = [random.uniform(GENE_MIN, GENE_MAX) for _ in range(num_genes)]
        population.append(individual)
    return population
population = generate_population(POPULATION_SIZE, NUM_GENES)
print(f"population:  {population}")
print("\n")


# Define the function to be optimized
def fitness_function(individual):
    # Example fitness function: sum of squares
    print(f"individual is: {individual}")
    print("\n")
    print(-sum(x**2 for x in individual), "in fitness_function")
    return -sum(x**2 for x in individual)


# Selection method: Tournament selection
def tournament_selection(population, fitness_function):
    tournament_size = 2
    selected = []
    for _ in range(len(population)):
        participants = random.sample(population, tournament_size)
        winner = max(participants, key=fitness_function)
        selected.append(winner)
    return selected
population = tournament_selection(population, fitness_function)

print("*"*20)
print(f"Population is: {population}")
print("\n")


# Crossover method: Single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Mutation method
def mutate(individual):
    mutated = []
    for gene in individual:
        if random.random() < MUTATION_RATE:
            mutated.append(random.uniform(GENE_MIN, GENE_MAX))
        else:
            mutated.append(gene)
    return mutated


next_generation = []
for _ in range(0, len(population), 2):
            parent1 = population[_]

            parent2 = population[_ + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
print(f"next_generation is: {next_generation}")
print("\n")




# Main genetic algorithm loop
def genetic_algorithm():
    population = generate_population(POPULATION_SIZE, NUM_GENES)
    generations = 0
    while True:
        population = tournament_selection(population, fitness_function)
        next_generation = []
        for _ in range(0, len(population), 2):
            parent1 = population[_]

            parent2 = population[_ + 1]
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        population = next_generation
        generations += 1
        best_individual = max(population, key=fitness_function)
        print("Target reached in {} generations.  individual: {}".format(generations, best_individual))
        if fitness_function(best_individual) >= TARGET:
            print("Target reached in {} generations. Best individual: {}".format(generations, best_individual))
        break

# Run the genetic algorithm
genetic_algorithm()            
            
            
            



