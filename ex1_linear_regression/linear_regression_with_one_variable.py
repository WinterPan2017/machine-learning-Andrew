import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Part1：从txt文件中读取数据，绘制成散点图
    f = open("ex1data1.txt", 'r')
    population = []
    profit = []
    for line in f.readlines():
        col1 = line.split(',')[0]
        col2 = line.split(',')[1].split('\n')[0]
        population.append(float(col1))
        profit.append(float(col2))
    plt.subplot(2, 2, 1)
    plt.title("Scatter plot of training data")
    plt.xlabel("population of city")
    plt.ylabel("profit")
    plt.scatter(population, profit, marker='x')

    # part2：递归下降，同时记录损失值的变化
    m = len(population)
    alpha = 0.01
    iterations = 1500
    theta = [0, 0]
    t = []
    cost = []
    theta0 = []
    theta1 = []
    c = 0
    for j in range(m):
        c += 1.0 / (2 * m) * pow(theta[0] + theta[1] * population[j] - profit[j], 2)
    print(c)
    for i in range(iterations):
        t.append(i)
        temp0 = theta[0]
        temp1 = theta[1]
        for j in range(m):
            temp0 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j])
            temp1 -= (alpha / m) * (theta[0] + theta[1] * population[j] - profit[j]) * population[j]
        theta[0] = temp0
        theta[1] = temp1
        c = 0
        for j in range(m):
            c += 1.0 / (2 * m) * pow(theta[0] + theta[1] * population[j] - profit[j], 2)
        cost.append(c)
        theta0.append(temp0)
        theta1.append(temp1)

    # part3：绘制回归直线图，已经损失函数变化图
    x = [5.0, 22.5]
    y = [5.0 * theta[1] + theta[0], 22.5 * theta[1] + theta[0]]
    plt.subplot(2, 2, 2)
    plt.plot(x, y, color="red")
    plt.title("Linear Regression")
    plt.xlabel("population of city")
    plt.ylabel("profit")
    plt.scatter(population, profit, marker='x')

    plt.subplot(2, 2, 3)
    plt.title("Visualizing J(θ)")
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.plot(t, cost, color="red")
    plt.show()

