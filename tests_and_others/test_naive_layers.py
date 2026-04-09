from layers import multiplication, addition


# multiplication
apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = multiplication()
mul_tax_layer = multiplication()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price)

# backward
d_price = 1
d_apple_price, d_tax = mul_tax_layer.backward(d_price)
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)

print(d_apple, d_apple_num, d_tax)

# addition
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = multiplication()
mul_orange_layer = multiplication()
add_apple_orange_layer = addition()
mul_tax_layer = multiplication()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)                 # 1
orange_price = mul_orange_layer.forward(orange, orange_num)             # 2
all_price = add_apple_orange_layer.forward(apple_price, orange_price)   # 3
price = mul_tax_layer.forward(all_price, tax)                           # 4

# backward
d_price = 1
d_all_price, d_tax = mul_tax_layer.backward(d_price)                      # 4
d_apple_price, d_orange_price = add_apple_orange_layer.backward(d_all_price)    # 3
d_orange, d_orange_num = mul_orange_layer.backward(d_orange_price)      # 2
d_apple, d_apple_num = mul_apple_layer.backward(d_apple_price)          # 1

print(price)
print(d_apple_num, d_apple, d_orange, d_orange_num, d_tax)