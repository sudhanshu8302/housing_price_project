from housing_price_project.static.model import model as md

print("Hello Developer", "", "Select an option-", sep="\n")
print("1. Visualize the Dataset\n2. Train the Model\n3. Evaluate the Model\n4. Predict")
choice = input("Enter:")

if choice == '1':
	md.Visualize_dataset()

if choice == '2':
	md.Train_model()

if choice == '3':
	md.Evaluate_model()

if choice == '4':
	area = input("Enter area:")
	rooms = input("Enter number of rooms:")
	price = md.Predict(area, rooms)
	print(f"Price = {price}")