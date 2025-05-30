#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include "readFile.cpp"
#include "mlp.cpp"
#include "matrixPrint.cpp"

/*
    Professor, o programa não está 100% funcionando, embora esteja tudo implementado corretamente, de acordo
    com os slides. Os pesos atualizados estão ficando bem próximos de 0.5, não sei porque. Fora que
    as saídas estão ficando muito parecidas (as f(net)). Para a base de dados iris, não achei parametros 
    que dessem uma boa resposta, na verdade, a base está sendo treinanda de forma muito ruim. Porém, nas outras
    bases está até que razoavel, muitos dos dados de teste retornam a resposta esperada. Acredito que tenha algo ainda de errado na lógica que esteja fazendo o algoritmo
    retornar saídas próximas à 0.5. Se o senhor achar o erro, por favor me manda uma mensagem, pois fiquei dias
    e dias olhando esse código, fazendo, refazendo, lendo papers e aplicando teorias de inicialização e normalização de dados
    para tentar melhorar a acurácia e os resultados, mas nada adiantou, o algoritmo ainda está meio ruim,mas funciona. 
    Mas enfim, eu aprendi muito vendo cada pedacinho desse código e desmembrando a rede aos pedaços e vendo
    como funciona o MLP. Temo que passei tanto tempo vendo esse código, que essa parte da matéria foi a que mais aprendi.
*/

using namespace std;

int main()
{

    // string file_data = "iris-data-3-class.txt";
    // string file_label = "iris-data-3-class-label.txt";

    // string file_data = "cancer-data.txt";
    // string file_label = "cancer-label.txt";
    // string file_class = "cancer-data-class.txt";

    string file_data = "vinho-data.txt";
    string file_label = "vinho-data-label.txt";
    string file_class = "vinho-data-class.txt";

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    Flower container = Flower();

    //aqui os dados são carregados
    container.data = readFile(file_data);
    container.label_string = readFileLabel(file_label);
    vector<string> class_data = readFileLabel(file_class);
   
    //normaliza os dados de entrada
    normalize(container,"vinho-data.txt");

    //transforma as labels em numeros (setosa -> 100, versicolor -> 010, virginica -> 001)
    container.fillOutputLabel();

    //embaralha os dados
    shuffle(container.data.begin(), container.data.end(), std::default_random_engine(seed));
    shuffle(container.label_output.begin(), container.label_output.end(), std::default_random_engine(seed));
    //entrada/camada escondida/processadores/saida/taxa de aprendizado
    ANN net = ANN(13, 1, 15, 3, 0.05);
    //faz as ligações dos neuronios
    net.buildANN();
    Solver s = Solver(net.graph.size());
    s.solveAll(net, container, 10000);
    //descomente essa linha se quiser visualizar as conexões dos neuronios
    //  net.showNetworkConnections();

    //coloque aqui o valor que deseja testar
    vector<double> input({12.88,2.99,2.4,20,104,1.3,1.22,.24,.83,5.4,.74,1.42,530});
    normalize(input,"vinho-data.txt");
    
    //encontra a resposta de acordo com uma determinada entrada
    input = s.responseFromNetwork(net, container, input);

    // ---------------------------------------------------
    // output para base de iris
    // cout << "resposta" << endl;
    // double maior=-10000;
    // double index=-1;
    // for (int i = 0; i < input.size(); i++)
    // {
    //     if(maior<input[i]){
    //         maior=input[i];
    //         index=i;
    //     }
        
    // }
    // cout<<class_data[index]<<endl;
    
    // ---------------------------------------------------
    // output para base de dados de vinho
    cout << "resposta" << endl;
    double maior=-10000;
    double index=-1;
    for (int i = 0; i < input.size(); i++)
    {
        if(maior<input[i]){
            maior=input[i];
            index=i;
        }
        
    }
    cout<<class_data[index]<<endl;

    //---------------------------------------------------
    //output para a base de cancer
    // cout << "resposta" << endl;
    // if(input[0]>=0.5)
    //     cout<<"M"<<endl;
    // else
    //     cout<<"B"<<endl;


    return 0;
}