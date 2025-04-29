import numpy as np
from sklearn.preprocessing import MinMaxScaler

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

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
        [-6.341,0.000,0.310,-1.34,-1.45],
        [-5.707,0.000,0.279,-1.37,-1.49],
        [-5.073,0.000,0.248,-1.40,-1.53], 
        [-4.439,0.000,0.217,-1.43,-1.58],
        [-3.805,0.000,0.186,-1.47,-1.62],
        [-3.171,0.000,0.155,-1.50,-1.66],
        [-2.536,0.000,0.124,-1.53,-1.70],
        [-1.902,0.000,0.093,-1.57,-1.74],
        [-1.268,0.000,0.062,-1.60,-1.78],
        [-0.634,0.000,0.031,-1.64,-1.82],
        [0.634,0.000,-0.031,-1.91,-1.71],
        [1.268,0.000,-0.062,-1.95,-1.74], 
        [1.902,0.000,-0.093,-1.99,-1.77], 
        [2.536,0.000,-0.124,-2.03,-1.81], 
        [3.171,0.000,-0.155,-2.07,-1.84], 
        [3.805,0.000,-0.186,-2.11,-1.88], 
        [4.439,0.000,-0.217,-2.15,-1.91], 
        [5.073,0.000,-0.248,-2.19,-1.95], 
        [5.707,0.000,-0.279,-2.23,-1.98], 
        [6.341,0.000,-0.310,-2.27,-2.02] 
    ])
    
    y = np.array([
        [-5.435,0.000,-0.250],
        [-5.067,0.000,-0.261],
        [-4.675,0.000,-0.276],
        [-4.257,0.000,-0.295],
        [-3.813,0.000,-0.318],
        [-3.344,0.000,-0.345],
        [-2.850,0.000,-0.376],
        [-2.329,0.000,-0.411],
        [-1.783,0.000,-0.450],
        [-1.210,0.000,-0.493],
        [0.013,0.000,-0.588],
        [0.664,0.000,-0.640],
        [1.342,0.000,-0.696],
        [2.045,0.000,-0.754],
        [2.775,0.000,-0.815],
        [3.531,0.000,-0.879],
        [4.313,0.000,-0.946],
        [5.122,0.000,-1.015],
        [5.956,0.000,-1.086],
        [6.816,0.000,-1.160]
    ])

    # 创建并训练网络
    model = DNNRegressor(
        input_size=5,
        hidden_sizes=(24, 16, 8),  # 增大首层容量
        learning_rate=0.0005,       # 调整学习率
        batch_size=8
    )
    model.train(X, y, epochs=30000, verbose=1000)
    
    # 测试预测
    test_input = np.array([[-3.505,0.000,0.230,-1.93,-1.71]])
    prediction = model.predict(test_input)
    print("\n测试输入:", test_input)
    print("预测结果:", prediction.round(3))
    print("实际目标:", y[0].round(3))