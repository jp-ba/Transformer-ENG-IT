class MyList:
    def __init__(self, data: any) -> None:
        self.data = data  # Store the data in an internal list

    def __getitem__(self, index: any) -> any:
        """
        Retrieves the item at the specified index using the [] operator.
        """
        print(f"Accessing item with index: {index} (type: {type(index).__name__})")
        return self.data[index] # Delegate the access to the internal list

# Create an object
my_list = MyList([1, 2, 3, 4, 5])

# Access elements using square brackets
print(f"my_list[2]: {my_list[2]}")
print(f"my_list[1:4]: {my_list[1:4]}")

#As shown in the output, when you use slicing [1:4], the index argument passed to __getitem__ is a slice object. 
#Common Uses
#Creating custom container types: Allows custom objects to behave like built-in containers (lists, dictionaries, etc.).
#Data Access: Frequently used in data analysis and machine learning frameworks, such as PyTorch's Dataset class, for accessing data samples.
#Lazy Loading: Can be used to load data dynamically only when it's accessed, rather than all at once.
#Iterability: When implemented with integer indices and a __len__ method, it makes an object iterable (usable in for loops).
