import numpy as np
import optometry as opt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def evaluate_model(model, X_test, y_test):
    # 预测
    y_pred = model.predict(X_test)
    
    # 反归一化
    y_true = y_test
   
    
    # 计算指标
    metrics = {
        'MSE': {
            'Per Feature': mean_squared_error(y_true, y_pred, multioutput='raw_values'),
            'Average': mean_squared_error(y_true, y_pred)
        },
        'MAE': {
            'Per Feature': mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
            'Average': mean_absolute_error(y_true, y_pred)
        },
        'R²': {
            'Per Feature': r2_score(y_true, y_pred, multioutput='raw_values'),
            'Average': r2_score(y_true, y_pred)
        },
        'Max Error': np.max(np.abs(y_true - y_pred), axis=0)
    }
    
    # 打印报告
    print(f"{'Metric':<15} | {'C(2,0)':>10} | {'C(2,-2)':>10} | {'C(2,2)':>10} | {'Average'}")
    print("-"*65)
    for metric in ['MSE', 'MAE', 'R²']:
        row = f"{metric:<15} | "
        for val in metrics[metric]['Per Feature']:
            if metric == 'R²':
                row += f"{val:>10.3f} | "
            else:
                row += f"{val:>10.4f} | "
        row += f"{metrics[metric]['Average']:>10.4f}"
        print(row)
    
    print("\nMax Errors:")
    print(f"C(2,0): {metrics['Max Error'][0]:.4f}")
    print(f"C(2,-2): {metrics['Max Error'][1]:.4f}") 
    print(f"C(2,2): {metrics['Max Error'][2]:.4f}")


class DNNRegressor:
    def __init__(self, 
                 input_size=5,
                 hidden_sizes=(16, 12, 8),
                 output_size=3,
                 learning_rate=0.001,
                 batch_size=5):
        # 网络结构初始化
        self.layers = []
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append({
                'W': np.random.randn(prev_size, size) * np.sqrt(2./prev_size),  # He初始化
                'b': np.zeros((1, size))
            })
            prev_size = size
        self.W_out = np.random.randn(prev_size, output_size) * 0.01
        self.b_out = np.zeros((1, output_size))
        
        # 训练参数
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def _forward(self, X):
        self.activations = [X]
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['W']) + layer['b']
            a = relu(z)
            self.activations.append(a)
        output = np.dot(self.activations[-1], self.W_out) + self.b_out
        return output

    def train(self, X, y, epochs=15000, verbose=1000):
        # 数据预处理
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 动量参数
        beta = 0.9
        v_W = [np.zeros_like(layer['W']) for layer in self.layers] + [np.zeros_like(self.W_out)]
        v_b = [np.zeros_like(layer['b']) for layer in self.layers] + [np.zeros_like(self.b_out)]
        
        for epoch in range(epochs):
            # 小批量训练
            indices = np.random.permutation(len(X_scaled))
            for i in range(0, len(X_scaled), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                X_batch = X_scaled[batch_idx]
                y_batch = y_scaled[batch_idx]
                
                # 前向传播
                output = self._forward(X_batch)
                
                # 计算损失
                loss = np.mean(0.5 * (y_batch - output)**2)
                
                # 反向传播修正
                grad = (output - y_batch) / self.batch_size
                
                # 存储各层梯度
                grads = []
                current_grad = grad
                
                # 反向计算梯度
                for i in reversed(range(len(self.layers))):
                    # 关键维度修正：先传递梯度再计算激活导数
                    grad_to_layer = np.dot(current_grad, self.W_out.T) if i == len(self.layers)-1 \
                                  else np.dot(current_grad, self.layers[i+1]['W'].T)
                    
                    delta = grad_to_layer * relu_derivative(self.activations[i+1])
                    grads.append(delta)
                    current_grad = delta
                
                # 参数更新（从输入层到输出层方向）
                grads = grads[::-1]  # 反转梯度列表
                for i in range(len(self.layers)):
                    delta = grads[i]
                    dW = np.dot(self.activations[i].T, delta)
                    db = np.sum(delta, axis=0, keepdims=True)
                    
                    # 动量更新
                    v_W[i] = beta * v_W[i] + (1 - beta) * dW
                    v_b[i] = beta * v_b[i] + (1 - beta) * db
                    self.layers[i]['W'] -= self.learning_rate * v_W[i]
                    self.layers[i]['b'] -= self.learning_rate * v_b[i]
                
                # 更新输出层
                dW_out = np.dot(self.activations[-1].T, grad)
                db_out = np.sum(grad, axis=0, keepdims=True)
                v_W[-1] = beta * v_W[-1] + (1 - beta) * dW_out
                v_b[-1] = beta * v_b[-1] + (1 - beta) * db_out
                self.W_out -= self.learning_rate * v_W[-1]
                self.b_out -= self.learning_rate * v_b[-1]
            
            if epoch % verbose == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        X_scaled = self.scaler_x.transform(X)
        output = self._forward(X_scaled)
        return self.scaler_y.inverse_transform(output)

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 输入数据（保持原始精度）
    X = np.array([

    [-5.435,0.000,-0.250,-1.34,-1.45,-15,0 ],
    [-5.067,0.000,-0.261,-1.37,-1.49,-15,0 ],
    [-4.675,0.000,-0.276,-1.40,-1.53,-15,0 ],
    [-4.257,0.000,-0.295,-1.43,-1.58,-15,0 ],
    [-3.813,0.000,-0.318,-1.47,-1.62,-15,0 ],
    [-3.344,0.000,-0.345,-1.50,-1.66,-15,0 ],
    [-2.850,0.000,-0.376,-1.53,-1.70,-15,0 ],
    [-2.329,0.000,-0.411,-1.57,-1.74,-15,0 ],
    [-1.783,0.000,-0.450,-1.60,-1.78,-15,0 ],
    [-1.210,0.000,-0.493,-1.64,-1.82,-15,0 ],
    [0.013,0.000,-0.588,-1.91,-1.71,-15,0 ],
    [0.664,0.000,-0.640,-1.95,-1.74,-15,0 ],
    [1.342,0.000,-0.696,-1.99,-1.77,-15,0 ],
    [2.045,0.000,-0.754,-2.03,-1.81,-15,0 ],
    [2.775,0.000,-0.815,-2.07,-1.84,-15,0 ],
    [3.531,0.000,-0.879,-2.11,-1.88,-15,0 ],
    [4.313,0.000,-0.946,-2.15,-1.91,-15,0 ],
    [5.122,0.000,-1.015,-2.19,-1.95,-15,0 ],
    [5.956,0.000,-1.086,-2.23,-1.98,-15,0 ],
    [6.816,0.000,-1.160,-2.27,-2.02,-15,0 ],
    [-3.505,0,0.230,-1.93,-1.71,0,15], 
    [-3.470,0,0.217,-1.89,-1.65,0,15 ], 
    [-3.377,0,0.213,-1.84,-1.60,0,15 ], 
    [-3.228,0,0.218,-1.79,-1.55,0,15 ], 
    [-3.021,0,0.231,-1.74,-1.50,0,15 ], 
    [-2.756,0,0.253,-1.69,-1.46,0,15],  
    [-2.433,0,0.284,-1.65,-1.42,0,15],  
    [-2.052,0,0.323,-1.60,-1.37,0,15],  
    [-1.613,0,0.369,-1.56,-1.33,0,15],  
    [-1.115,0,0.424,-1.52,-1.30,0,15],  
    [0.058,0,0.555,-1.23,-1.44,0,15],  
    [0.732,0,0.632,-1.20,-1.41,0,15],  
    [1.466,0,0.716,-1.17,-1.37,0,15],  
    [2.259,0,0.806,-1.14,-1.34,0,15],  
    [3.112,0,0.903,-1.11,-1.31,0,15],  
    [4.024,0,1.006,-1.09,-1.27,0,15],  
    [4.995,0,1.116,-1.06,-1.24,0,15],  
    [6.026,0,1.231,-1.04,-1.22,0,15],  
    [7.116,0,1.353,-1.01,-1.19,0,15],  
    [8.266,0,1.481,-0.99,-1.16,0,15] 
])

    y = np.array([
        [-6.341,0.000,0.310],
        [-5.707,0.000,0.279 ],
        [-5.073,0.000,0.248], 
        [-4.439,0.000,0.217],
        [-3.805,0.000,0.186],
        [-3.171,0.000,0.155],
        [-2.536,0.000,0.124],
        [-1.902,0.000,0.093],
        [-1.268,0.000,0.062],
        [-0.634,0.000,0.031],
        [0.634,0.000,-0.031],
        [1.268,0.000,-0.062], 
        [1.902,0.000,-0.093], 
        [2.536,0.000,-0.124], 
        [3.171,0.000,-0.155], 
        [3.805,0.000,-0.186], 
        [4.439,0.000,-0.217], 
        [5.073,0.000,-0.248], 
        [5.707,0.000,-0.279], 
        [6.341,0.000,-0.310], 
        [-6.341,0,-0.310], 
        [-5.707,0,-0.279],  
        [-5.073,0,-0.248],  
        [-4.439,0,-0.217], 
        [-3.805,0,-0.186],  
        [-3.171,0,-0.155],  
        [-2.536,0,-0.124],  
        [-2.536,0,-0.124],  
        [-1.268,0,-0.062],  
        [-0.634,0,-0.031],  
        [0.634,0,0.031],  
        [1.268,0,0.062],  
        [1.902,0,0.093],  
        [2.536,0,0.124],  
        [3.171,0,0.155],  
        [3.805,0,0.186],  
        [4.439,0,0.217],  
        [5.073,0,0.248],  
        [5.707,0,0.279],  
        [6.341,0,0.310]
    ])
    # 创建并训练网络
    model = DNNRegressor(
        input_size=7,
        hidden_sizes=(24, 16, 8),  # 增大首层容量
        learning_rate=0.0005,       # 调整学习率
        batch_size=8
    )
    model.train(X, y, epochs=100000, verbose=1000)
    
    X_t = np.array([
    [-5.671,0.013,0.638,-1.34,-1.45,-15,0],
    [-5.177,-0.162,0.447,-1.37,-1.49,-15,0],
    [-4.557,-0.306,0.348,-1.4,-1.53,-15,0],
    [-3.758,0,0.408,-1.43,-1.58,-15,0],
    [-3.089,-0.082,0.342,-1.47,-1.62,-15,0],
    [-2.617,-0.086,0.17,-1.5,-1.66,-15,0],
    [-2.095,-0.138,0.119,-1.53,-1.7,-15,0],
    [-1.371,0.004,0.173,-1.57,-1.74,-15,0],
    [-1.147,-0.165,-0.052,-1.6,-1.78,-15,0],
    [-0.673,-0.021,-0.103,-1.64,-1.82,-15,0],
    [0.414,0.008,-0.044,-1.91,-1.71,-15,0],
    [0.825,-0.006,-0.081,-1.95,-1.74,-15,0],
    [1.229,-0.004,-0.125,-1.99,-1.77,-15,0],
    [1.569,0,-0.168,-2.03,-1.81,-15,0],
    [1.961,0,-0.185,-2.07,-1.84,-15,0],
    [2.162,0.086,-0.354,-2.11,-1.88,-15,0],
    [2.63,-0.109,-0.226,-2.15,-1.91,-15,0],
    [2.837,-0.043,-0.39,-2.19,-1.95,-15,0],
    [3.308,0.097,-0.277,-2.23,-1.98,-15,0],
    [3.574,0.105,-0.286,-2.27,-2.02,-15,0]

])
    

    # 测试预测
    #test_input = X
    test_input = X_t
    prediction = model.predict(test_input)
    #print("\n测试输入:", test_input)
    #print("预测结果:", prediction.round(3))
    #print("实际目标:", y.round(3))
    
    for res in prediction:
        zlist = np.zeros(6)
        zlist[3] = res[1]
        zlist[4] = res[0]
        zlist[5] = res[2]
        M,J45, J180 = opt.calc_M_J45_J180(zlist,2,0,-15)
        cyl,sph,theta = opt.calc_sph_cyl_theta(M,J45,J180)
        #print(res )
        #print(' M,J45, J180:', M,J45, J180 )
        print('cyl,sph,theta:',cyl,sph,theta)
'''
    evaluate_model(model,X,y)

    with open("train_res.csv", "w", newline="") as f:
        f.write("# --- Z(2,0) Z(2,-2) Z(2,2) ---\n")  # 添加分隔注释
        f.write("# --- input data ---\n")  # 添加分隔注释
        np.savetxt(f, test_input, delimiter=",", fmt="%.3f")
        f.write("# --- predict res ---\n")  # 添加分隔注释
        np.savetxt(f, prediction.round(3), delimiter=",", fmt="%.3f")
        f.write("# --- true data ---\n")  # 添加分隔注释
        np.savetxt(f, y.round(3), delimiter=",", fmt="%.3f")

'''