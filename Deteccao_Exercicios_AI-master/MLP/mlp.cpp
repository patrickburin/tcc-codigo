#include <iostream>
#include <cstdlib>
#include <vector>
#include <time.h>
#include <list>

using namespace std;

class Flower
{
public:
    vector<vector<double>> data;
    vector<string> label_string;
    vector<vector<double>> label_output;
    vector<double> max;
    vector<double> min;

    //transforma as labels em código binário para cada classe
    void fillOutputLabel()
    {
        for (int i = 0; i < this->label_string.size(); i++)
        {
            //se a base é iris, descomente esses if e comente o restante
            // if (this->label_string[i] == "Iris-setosa")
            // {
            //     this->label_output.push_back({1, 0, 0});
            // }
            // else if (this->label_string[i] == "Iris-versicolor")
            // {
            //     this->label_output.push_back({0, 1, 0});
            // }
            // else if (this->label_string[i] == "Iris-virginica")
            // {
            //     this->label_output.push_back({0, 0, 1});
            // }

            //se a base é cancer, descomente esses if e comente o restante
            // if (this->label_string[i] == "M")
            // {
            //     this->label_output.push_back({1});
            // }
            // else if (this->label_string[i] == "B")
            // {
            //     this->label_output.push_back({0});
            // }

            //se a base é a de vinho, descomente esses if e comente o restante
            if (this->label_string[i] == "1")
            {
                this->label_output.push_back({1, 0, 0});
            }
            else if (this->label_string[i] == "2")
            {
                this->label_output.push_back({0, 1, 0});
            }
            else if (this->label_string[i] == "3")
            {
                this->label_output.push_back({0, 0, 1});
            }
           
        }
    }
};

//aqui é feita a normalização dos dados
void normalize(Flower &data, string f)
{
    string file_data = f;
    Flower container = Flower();
    container.data = readFile(file_data);

    vector<double> max(container.data[0].size(), -10000.0);
    vector<double> min(container.data[0].size(), 10000.0);

    for (int i = 0; i < container.data[0].size(); i++)
    {
        for (int j = 0; j < container.data.size(); j++)
        {
            

            if (max[i] < container.data[j][i])
            {
                max[i] = container.data[j][i];
            }
            if (min[i] > container.data[j][i])
            {
                min[i] = container.data[j][i];
            }
        }
    }
    for (int i = 0; i < data.data.size(); i++)
    {
        for (int j = 0; j < data.data[i].size(); j++)
        {
            double num = (max[j] - min[j]);
            
            if(num==0){
                num=1;    
            }
            
            data.data[i][j] = (data.data[i][j] - min[j]) / num;
            
        }
    }
}
//sobrecarga da função de normalização para o dado de teste
void normalize(vector<double> &data,string f)
{
    string file_data = f;
    Flower container = Flower();
    container.data = readFile(file_data);

    vector<double> max(container.data[0].size(), -10000.0);
    vector<double> min(container.data[0].size(), 10000.0);

    for (int i = 0; i < container.data[0].size(); i++)
    {
        for (int j = 0; j < container.data.size(); j++)
        {
            if (max[i] < container.data[j][i])
            {
                max[i] = container.data[j][i];
            }
            if (min[i] > container.data[j][i])
            {
                min[i] = container.data[j][i];
            }
        }
    }
    for (int i = 0; i < data.size(); i++)
    {
        double num = (max[i] - min[i]);
            
            if(num==0){
                num=1;    
            }

            data[i] = (data[i] - min[i]) / num;
        
    }
}
//classe que armazena e constrói a estrutura em grafo da rede neural
class ANN
{
public:
    int input;
    int hidden;
    int processors;
    int output;
    double n;
    vector<list<int>> graph;
    vector<vector<double>> weights;

    ANN(int input, int hidden, int processors, int output, double n)
    {
        this->input = input;
        this->hidden = hidden;
        this->processors = processors;
        this->output = output;
        this->n = n;
        vector<list<int>> aux(input + output + (hidden * processors));
        vector<double> v1(input + output + (hidden * processors), 0);
        vector<vector<double>> ax(input + output + (hidden * processors), v1);
        this->weights = ax;
        this->graph = aux;
    }
    //aqui é feita a lógica para fazer a ligação da rede
    //a estrutura é a seguinte
    /*
    
    dado uma rede com 4 entrada, uma camada escondida com 4 processadores e dois 
    neuronios na camada de saida, temos a seguinte ligação

    0  4
    1  5  8
    2  6  9
    3  7
    
    */
    
    void buildANN()
    {
        int cont = 0;
        srand(time(NULL));
        //adicionando primeiro as entradas
        for (int i = 0; i < this->input; i++)
        {

            for (int j = this->input; j < this->input + this->processors; j++)
            {
                this->graph[i].push_back(j);
                this->weights[i][j] = (rand() % 999 + 1) / 1000.0;
            }
        }
        //adicionando ao grapo o restante dos nós
        for (int i = this->input; i < this->graph.size() - this->output; i++)
        {
            if ((i - this->input) % this->processors == 0)
                cont = 0;

            for (int j = i + this->processors - cont; j < i + this->processors - cont + this->processors; j++)
            {
                if (j < this->graph.size())
                {
                    this->graph[i].push_back(j);
                    this->weights[i][j] = (rand() % 999 + 1) / 1000.0;
                }
            }
            cont++;
        }
    }
    //esse método mostra como a rede em grafo está conectada
    void showNetworkConnections()
    {

        list<int>::iterator it;

        for (int i = 0; i < this->graph.size(); i++)
        {
            cout << i << " estah conectado a: ";
            for (it = this->graph[i].begin(); it != this->graph[i].end(); it++)
            {
                cout << *it << " ";
            }
            cout << endl;
        }

        for (int i = 0; i < this->weights.size(); i++)
        {
            for (int j = 0; j < this->weights[i].size(); j++)
            {
                cout << this->weights[i][j] << " ";
            }
            cout << endl;
        }
    }
};

//essa classe é responsável por fazer o forward e backpropagation
class Solver
{
public:
    vector<double> fnet;//saidas
    vector<double> output;//vetor de auxilio (não foi utilizado)
    vector<double> err;//vetor que guarda os erros de cada neuronio

    //inicialização dos vetores
    Solver(int tam)
    {
        vector<double> v1(tam, 0);
        vector<double> v2(tam, 0);
        vector<double> v3(tam, 0);
        this->fnet = v1;
        this->output = v2;
        this->err = v3;
    }
    //método responsável por retornar a resposta da classificação
    vector<double> responseFromNetwork(ANN g, Flower dt, vector<double> input)
    {
        for (int i = 0; i < this->fnet.size(); i++)
        {
            this->fnet[i] = 0;
        }
        for (int i = 0; i < input.size(); i++)
        {
            this->fnet[i] = input[i];
        }

        for (int c = 0; c < g.graph.size(); c++)
        {
            if (c >= g.input)
            {

                if ((c - g.input) % g.processors == 0 && c != g.graph.size() - g.output)
                {
                    for (int y = c; y < c + g.processors; y++)
                    {
                        if (y < this->fnet.size())
                            this->fnet[y] = FNET(this->fnet[y]);
                    }
                }
                else if (c == g.graph.size() - g.output)
                {

                    for (int y = c; y < g.graph.size(); y++)
                    {

                        this->fnet[y] = FNET(this->fnet[y]);
                    }
                }
            }

            list<int>::iterator it;
            for (it = g.graph[c].begin(); it != g.graph[c].end(); it++)
            {
                this->fnet[*it] += g.weights[c][*it] * this->fnet[c];
                
            }
        }

        vector<double> aux;

        for (int i = g.graph.size() - g.output; i < g.graph.size(); i++)
        {
            aux.push_back(this->fnet[i]);
        }
        return aux;
    }

    double FNET(double num)
    {
        return 1 / (1 + exp(-num));
    }
    //função que resolve tudo (forward e back-propagation)
    void solveAll(ANN &g, Flower dt, int iteration)
    {

        for (int itr = 0; itr < iteration; itr++)
        {
            for (int i = 0; i < dt.data.size(); i++)
            {
                for (int j = 0; j < g.graph.size(); j++)
                {
                    this->fnet[j] = 0;
                    this->err[j] = 0;
                }

                for (int j = 0; j < dt.data[i].size(); j++)
                {
                    this->fnet[j] = dt.data[i][j];
                }

                //primeiro é calculado todas as saídas
                for (int c = 0; c < g.graph.size(); c++)
                {
                    if (c >= g.input)
                    {
                        //essa lógica é responsável por verificar quando o calculo foi para a próxima
                        //camada, ai quando foi, ele faz o fnet para as saidas
                        if ((c - g.input) % g.processors == 0 && c != g.graph.size() - g.output)
                        {
                        
                            for (int y = c; y < c + g.processors; y++)
                            {

                                if (y < this->fnet.size())
                                {
                                    this->fnet[y] = FNET(this->fnet[y]);
                                }
                            }
                        }
                        else if (c == g.graph.size() - g.output)
                        {
                            
                            for (int y = c; y < g.graph.size(); y++)
                            {
                                this->fnet[y] = FNET(this->fnet[y]);
                            }
                        }
                    }
                    //for para fazer a soma dos pesos vezes a entrada
                    list<int>::iterator it;
                    for (it = g.graph[c].begin(); it != g.graph[c].end(); it++)
                    {
                        this->fnet[*it] += g.weights[c][*it] * this->fnet[c];
                    }
                }
               
                //calcula o erro apenas na saida
                for (int c = g.graph.size() - g.output; c < g.graph.size(); c++)
                {
                    this->err[c] = (dt.label_output[i][c - (g.graph.size() - g.output)] - this->fnet[c]) * this->fnet[c] * (1 - this->fnet[c]);
                }

                //erro do restante da rede
                for (int c = g.graph.size() - g.output - 1; c >= g.input; c--)
                {
                    list<int>::iterator it;
                    double soma = 0;

                    for (it = g.graph[c].begin(); it != g.graph[c].end(); it++)
                    {
                        soma += this->err[*it] * g.weights[c][*it];
                    }
                    this->err[c] = soma * this->fnet[c] * (1 - this->fnet[c]);
                }
                //aqui todos os pesos são atualizados, apenas quando todos os erros são calculados
                for (int c = 0; c < g.graph.size() - g.output; c++)
                {
                    list<int>::iterator it;
                    for (it = g.graph[c].begin(); it != g.graph[c].end(); it++)
                    {
                        g.weights[c][*it] = g.n * this->err[*it] * this->fnet[c];
                    }
                }
            }
        }
    }
};