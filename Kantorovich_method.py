import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def func(x, y):
    return y**2/x


def solution(x, C):
    return -1/(math.log(x)+C)


def euclid(d1, d2):
    return (d1 ** 2 + d2 ** 2)


def euclid_vect(v):
    _sum = 0
    for i in v:
        _sum += i**2
    return _sum


def f(y, z, x):
    return y**2


def g(y, z, x):
    return z**2


def h(y, z, x):
    return y*np.sin(math.pi*x)


class Solver_Poisson:
    def __init__(self, f, x0, x_left, x_right, y_left, y_right, step):
        self.f2 = f
        self.x0 = x0
        self.x_left = x_left
        self.x_right = x_right
        self.y_left = y_left
        self.y_right = y_right
        self.step = step

    def f1(self, x, y, z):
        return z

    def step_forward(self, x, yz, f2, h):
        y, z = yz
        k1_y = self.f1(x, y, z)
        k1_z = f2(x, y, z)
        k2_y = self.f1(x + h/2, y + k1_y*h/2, z + k1_z*h/2)
        k2_z = f2(x + h/2, y + k1_y*h/2, z + k1_z*h/2)
        k3_y = self.f1(x + h/2, y + k2_y*h/2, z + k2_z*h/2)
        k3_z = f2(x + h/2, y + k2_y*h/2, z + k2_z*h/2)
        k4_y = self.f1(x + h, y + k3_y*h, z + k3_z*h)
        k4_z = f2(x + h, y + k3_y*h, z + k3_z*h)
        y_new = y + h*(k1_y + 2*k2_y + 2*k3_y + k4_y)/6
        z_new = z + h*(k1_z + 2*k2_z + 2*k3_z + k4_z)/6
        return y_new, z_new

    def left_step(self, x, yz, f):
        return self.step_forward(x, yz, f, -self.step)

    def right_step(self, x, yz, f):
        return self.step_forward(x, yz, f, self.step)

    # TODO: rewrite with lambda
    def shot(self, x, yz, direction, f, need_values=False):
        y, z = yz
        values_y = []
        values_z = []
        if need_values:
            values_y.append(y)
            values_z.append(z)
        test = (x >= self.x_left) if direction == -1 else (x <= self.x_right)
        while (test):
            y, z = self.left_step(x, (y, z), f) if direction == -1 else self.right_step(x, (y, z), f)
            x += self.step*direction
            if need_values:
                values_y.append(y)
                values_z.append(z)
            test = (x >= self.x_left) if direction == -1 else (x <= self.x_right)

        return y, values_y, values_z

    def left_shot(self, x, yz, f, need_values=False):
        return self.shot(x, yz, -1, f, need_values)

    def right_shot(self, x, yz, f, need_values=False):
        return self.shot(x, yz, 1, f, need_values)

    def both_sides_shot(self, x, yz, f):
        _, left_array_y, left_array_z = self.left_shot(x, yz, f, True)
        _, right_array_y, right_array_z = self.right_shot(x, yz, f, True)
        left_array_y.reverse()
        left_array_z.reverse()
        # NOTE: fix join sides
        all_array_y = left_array_y+right_array_y[1:]
        all_array_z = left_array_z+right_array_z[1:]
        return all_array_y, all_array_z

    def discrep_2(self, v, f):
        yz = v[0], v[1]
        y_right_res, _, _ = self.right_shot(self.x0, yz, f)
        y_left_res, _, _ = self.left_shot(self.x0, yz, f)
        # NOTE: fix join sides
        right_discrep = self.y_right - y_right_res
        left_discrep = self.y_left - y_left_res
        return np.array([left_discrep, right_discrep])

    def deriv_vect2(self, v, f, tau):
        y0, z0 = v[0], v[1]
        d1y = self.discrep_2([y0 - tau / 2, z0], f)
        d2y = self.discrep_2([y0 + tau / 2, z0], f)
        d1z = self.discrep_2([y0, z0 - tau / 2], f)
        d2z = self.discrep_2([y0, z0 + tau / 2], f)
        return np.array([(d2y - d1y)/tau, (d2z - d1z)/tau])

    def Newton_vect(self, v_initial, f, eps):
        d = self.discrep_2(v_initial, f)
        v = v_initial
        counter = 0
        while (euclid_vect(d) >= eps):
            counter += 1
            print(counter)
            print(euclid_vect(d))
            m = np.transpose(self.deriv_vect2(v, f, self.step))
            v -= np.dot(np.linalg.inv(m), d)
            d = self.discrep_2(v, f)
        return v

    def integr(self, arr1, arr2, f, step=0):
        if (step == 0):
            step = self.step
        s = 0
        le = len(arr1)

        for i in range(le):
            s += f(arr1[i], arr2[i], self.x_left + i*step)
        s -= (f(arr1[0], arr2[0], self.x_left) + f(arr1[le-1], arr2[le-1], self.x_left + (le-1)*step)) / 2
        s *= step
        return s

    def calc_coeff(self, arr_y, arr_z, f, g, h):
        W1_res = self.integr(arr_y, arr_z, f)
        W2_res = self.integr(arr_y, arr_z, g)
        W3_res = self.integr(arr_y, arr_z, h)
        return W1_res, W2_res, W3_res

    def test_coeff(self, C1, C2, C3, C1_new, C2_new, C3_new, eps):
        changed = False
        changed = changed or self.test_one_coeff(C1, C1_new, eps)
        changed = changed or self.test_one_coeff(C2, C2_new, eps)
        changed = changed or self.test_one_coeff(C3, C3_new, eps)
        return changed

    def test_one_coeff(self, C1, C1_new, eps):
        # Note: devide on C1 cause we want to get big values
        return abs((C1-C1_new)/C1) > eps

    def deriv_kant_vect(self, W_vect, tau):
        W1, W2 = W_vect[0], W_vect[1]
        d1W1 = self.discrep_kant([W1 - tau / 2, W2])
        d2W1 = self.discrep_kant([W1 + tau / 2, W2])
        d1W2 = self.discrep_kant([W1, W2 - tau / 2])
        d2W2 = self.discrep_kant([W1, W2 + tau / 2])
        return np.array([(d2W1 - d1W1)/tau, (d2W2 - d1W2)/tau])

    def result_kant(self, W_vect):
        return self.discrep_kant(W_vect, True)

    def discrep_kant(self, W_vect, result=False):
        W1, W2 = W_vect[0], W_vect[1]
        y = 1
        z = 1

        def ODE_V(x, y, z):
            return W1*y + W2*np.sin(math.pi*x)

        y_v, z_v = self.Newton_vect([y, z], ODE_V, 0.001)
        arr_y_v, arr_z_v = self.both_sides_shot(self.x0, (y_v, z_v), ODE_V)
        V_res, VP_res, VS_res = self.calc_coeff(arr_y_v, arr_z_v, f, g, h)
        V1_res = VP_res / V_res
        V2_res = VS_res / V_res

        def ODE_W(x, y, z):
            return V1_res*y + V2_res*np.sin(math.pi*x)

        y_w, z_w = self.Newton_vect([y, z], ODE_W, 0.001)
        arr_y_w, arr_z_w = self.both_sides_shot(self.x0, (y_w, z_w), ODE_W)
        W_res, WP_res, WS_res = self.calc_coeff(arr_y_w, arr_z_w, f, g, h)
        W1_res = WP_res / W_res
        W2_res = WS_res / W_res

        if result:
            # NOTE: fix join sides
            return arr_y_v, arr_y_w[1:]
        return np.array([W1 - W1_res, W2 - W2_res])

    def Newton_kant_vect(self, W_initial, eps):
        d = self.discrep_kant(W_initial)
        W_vect = W_initial
        while (euclid_vect(d) >= eps):
            m = np.transpose(self.deriv_kant_vect(W_vect, self.step))
            W_vect -= np.dot(np.linalg.inv(m), d)
            d = self.discrep_kant(W_vect)
        return W_vect

    def Kantorovich(self, f, g, h, eps):
        W, WP, WS = 1, 1, 1
        V, VP, VS = -1, 1, 1
        y = 1
        z = 1

        should_continue = True
        while (should_continue):
            def ODE_W(x, y, z):
                return (VP*y + VS*np.sin(math.pi*x)) / V

            def ODE_V(x, y, z):
                return (WP*y + WS*np.sin(math.pi*x)) / W

            y_v, z_v = self.Newton_vect([y, z], ODE_V, 0.001)
            y_w, z_w = self.Newton_vect([y, z], ODE_W, 0.001)
            arr_y_v, arr_z_v = self.both_sides_shot(self.x0, (y_v, z_v), ODE_V)
            arr_y_w, arr_z_w = self.both_sides_shot(self.x0, (y_w, z_w), ODE_W)
            W_res, WP_res, WS_res = self.calc_coeff(arr_y_w, arr_z_w, f, g, h)
            V_res, VP_res, VS_res = self.calc_coeff(arr_y_w, arr_z_v, f, g, h)

            should_continue = self.test_coeff(W_res, WP_res, WS_res, W, WP, WS,  eps)
            should_continue = should_continue or self.test_coeff(V_res, VP_res, VS_res, V, VP, VS,  eps)
            W, WP, WS = W_res, WP_res, WS_res
            V, VP, VS = V_res, VP_res, VS_res

            # NOTE: for check
            print(W, WP, WS)
            print(V, VP, VS)
            print('----------------')

        return arr_y_v, arr_y_w


# Run example
if __name__ == "__main__":
    # initial values
    beta = 0.5
    omg = 5
    x0 = 0.5
    x_left = 0
    x_right = 1
    y0 = 0.2
    y_left = 0
    y_right = 0
    y_prime = 0.5
    step = 0.02

    # some function
    def func_(x, y, y_prime):
        return -beta*y_prime-(omg ** 2)*y

    # initialize class
    sol_p = Solver_Poisson(f=func_, x0=x0, x_left=x_left, x_right=x_right,
                           y_left=y_left, y_right=y_right, step=step)

    def func_to_plot(x, y):
        '''func to plot numerical solution'''
        lx = len(kant_res[0])
        ly = len(kant_res[1])
        nx = int(x*(lx-1))
        ny = int(y*(ly-1))
        return kant_res[0][nx] * kant_res[1][ny]

    def analytical_solution(x, y):
        '''func to plot analytical solution'''
        return (- 1 / (2 * (math.pi**2)))*math.sin(math.pi*x)*math.sin(math.pi*y)

    # kant_res = sol_p.Kantorovich(f=f, g=g, h=h, eps=0.05)
    W_res = sol_p.Newton_kant_vect([1, 1], eps=0.05)
    kant_res = sol_p.result_kant(W_res)

    # init indexes
    xs = [x for x in np.arange(0, 1.02, 0.02)]
    ys = [x for x in np.arange(0, 1.02, 0.02)]
    # functions results
    zs1 = [analytical_solution(x, y) for x, y in zip(xs, ys)]
    zs2 = [func_to_plot(x, y) for x, y in zip(xs, ys)]

    # plot_1d
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xs, ys, zs1, linewidth=3, label='analytic curve')
    ax.plot(xs, ys, zs2, label='kantarovich curve')
    ax.legend()
    plt.show()

    # plot_3d
    zs_anlt = [[analytical_solution(x, y) for y in ys] for x in xs]
    zs_kant = [[func_to_plot(x, y) for y in ys] for x in xs]
    xs = np.array(xs)
    ys = np.array(ys)
    zs_anlt = np.array(zs_anlt)
    zs_kant = np.array(zs_kant)

    # plot_3d analitycal
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, zs_anlt, antialiased=True, label='Analytical curve')
    ax.text(0, 1, 0, 'Analytical solution', color='red')
    plt.show()

    # plot_3d numerical
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, zs_kant, cmap=cm.viridis, label='Kantorovich curve')
    ax.text(0, 1, 0, 'Kantorovich solution', color='red')
    plt.show()

    # compare plot_3d
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    X, Y = np.meshgrid(xs, ys)
    plot_1 = ax.plot_surface(X, Y, zs_anlt, antialiased=True, label='Analytical curve')
    plot_1._facecolors2d = plot_1._facecolors3d
    plot_1._edgecolors2d = plot_1._edgecolors3d
    plot_2 = ax.plot_surface(X, Y, zs_kant, antialiased=True, label='Kantorovich curve', alpha=0.6)
    plot_2._facecolors2d = plot_2._facecolors3d
    plot_2._edgecolors2d = plot_2._edgecolors3d
    ax.legend()
    plt.show()
