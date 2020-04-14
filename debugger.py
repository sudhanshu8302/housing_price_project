from housing_price_project.static.model import model as md

print("Hello Developer", "", "Select an option-", sep="\n")

while True:
	print("1. Visualize the Dataset\n2. Train the Model\n3. Evaluate the Model\n4. Predict\n5.Exit")
	choice = input("Enter:")

	if choice == '1':
		md.Visualize_dataset()

	if choice == '2':
		md.Train_model()

	if choice == '3':
		md.Evaluate_model()

	if choice == '4':
		Propertycount = input("Enter propertycount:")
		Rooms = input("Enter number of rooms:")
		Type = input("Enter type:")
		Distance = input("Enter distance:")
		price = md.Predict(Rooms, Type, Propertycount, Distance)
		print(f"Price = {price}")
		input("Press any key to continue!\n")
	if choice =='5':
		break
print("BYE")

