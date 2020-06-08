#include<fstream>
#include<iostream>
#include<map>
#include<vector>
#include<algorithm>
#include<cmath>
#include <boost/python/numpy.hpp>
#include <Python.h>

#include <GL/glut.h>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

struct event{
	event(int idx, double seq, int type): idx(idx), seq(seq), type(type){}
	event(int idx, double seq, int type, double value): idx(idx), seq(seq), type(type), value(value){}
	int idx;
	double seq;
	int type; // 0~7 : key, 8: bpm, 9: stop
	double value;
	bool operator<(const event &l)const{
		return seq != l.seq ? seq < l.seq : type < l.type;
	}
};

struct notes{
	notes(){for(int i = 0; i < 8; i++) note[i] = 0;}
	double note[8];
	double& operator [](const int &l){ return note[l]; }
};

struct Pattern{
	vector<notes> data;
	double period;
	void add_note(double time, int key){
		int t = (time + 2.5) / period;
		for(int i = -3; i <= 3; i++){
			while((int)data.size() <= i+t) data.push_back(notes());
			double start = (i+t-0.5)*period - (2.5 + time), end = (i+t+0.5)*period - (2.5 + time);
			double value = get_weight(start, end);
			data[i+t][key] += value;
		}
	}
	double get_weight(double start, double end){
		const double W = 0.05, S = 1/W/W;
		start = min(max(start, -W), W);
		end = max(min(end, W), -W);
		auto f = [&](double x){ return x < 0? S*(x+W)*(x+W)/2 : 1.0 - S*(W-x)*(W-x)/2; };
		return f(end) - f(start);
	}
};

int hex_to_int(string hex){
	int v = 0;
	for(char c : hex) v = v * 16 + (c >= 'a'? c-'a'+10 : c >= 'A'? c-'A'+10 : c - '0');
	return v;
}

Pattern pattern_parser(std::string pattern_name, double period)
{
	Pattern pattern;
	pattern.period = period;
	
	vector<event> L;

	ifstream in(pattern_name);
	if(!in.is_open()){
		cout << "pattern " << pattern_name << " not found.\n";
		return pattern;
	}
	string line;
	double bpm = 130;
	map<string, double> bpm_log, stop_log;
	double seq_multiplier[1000] = {};
	for(int i = 0; i < 1000; i++) seq_multiplier[i] = 1;
	while(getline(in, line)){
//		std::cout << line << "\n";
		if(line.substr(0, 5) == "#BPM ") bpm = stod(line.substr(5, string::npos));
		else if(line.substr(0, 4) == "#BPM") bpm_log[line.substr(4, 2)] = stod(line.substr(7, string::npos));
		else if(line.substr(0, 5) == "#STOP") stop_log[line.substr(5, 2)] = stod(line.substr(8, string::npos));
		if(line.size() <= 6 || line[6] != ':' || line[1] > '9' || line[1] < '0') continue;

		int seq = stoi(line.substr(1, 3));
		int channel = hex_to_int(line.substr(4, 2));
		string data = line.substr(7, string::npos);
		int len = data.size() / 2;
		if(channel == 2){
			seq_multiplier[seq] = stod(data);
			L.emplace_back(seq, seq, -1, -1);
		} else if(channel == 3){
			for(int i = 0; i < len; i++){
				string value = data.substr(i*2, 2);
				if(value != "00"){
					double bpm;
					if(bpm_log.find(value) != bpm_log.end()) bpm = bpm_log[value];
					else bpm = hex_to_int(value);
					L.emplace_back(seq, seq + i / (double)len, 8, bpm);
				}
			}
		} else if(channel == 8){
			for(int i = 0; i < len; i++){
				string value = data.substr(i*2, 2);
				if(value != "00"){
					double stop;
					if(stop_log.find(value) != stop_log.end()) stop = stop_log[value];
					else stop = hex_to_int(value);
					L.emplace_back(seq, seq + i / (double)len, 9, stop);
				}
			}
		} else if(channel/16 == 1){
			for(int i = 0; i < len; i++){
				int value = hex_to_int(data.substr(i*2, 2));
				if(value != 0){
					int tmp = channel%16;
					int DB[] = {-1, 1, 2, 3, 4, 5, 0, -1, 6, 7};
					if(tmp >= 10) printf("???\n");
					L.emplace_back(seq, seq + i / (double)len, DB[tmp]);
				}
			}
		}
	}
	double last_seq = 0, last_time = 0;
	int last_idx = 0;
	sort(L.begin(), L.end());
	for(event e : L){
//		printf("%lf %lf\n", e.seq, last_time);
		last_time += (e.seq - last_seq) * 240.0 / bpm * seq_multiplier[last_idx];
		last_seq = e.seq;
		last_idx = e.idx;
		if(e.type == 8){
			bpm = e.value;
		} else if(e.type == 9){
			last_time += 240.0 / bpm / 192 * e.value;
		} else if(0 <= e.type && e.type <= 7){
			pattern.add_note(last_time, e.type);
		}
	}
//	printf("bpm : %lf\n", bpm);
//	printf("last seq : %lf\n", L.back().seq);
///	printf("song total length : %lf\n", last_time);

	in.close();
	return pattern;
}

np::ndarray pattern_parser_python(std::string name, double period){
//	std::cout << "name : " << name << "\n";
	Pattern pattern = pattern_parser(name, period);
	int r = (int)pattern.data.size(), c = 8;
	p::tuple shape = p::make_tuple(r, c);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray np = np::empty(shape, dtype);
	for(int i = 0; i < r; i++)
		for(int j = 0; j < c; j++)
				np[i][j] = pattern.data[i][j];
	return np; 
}

int step = 0;
Pattern pattern;

void display(){
	glClear(GL_COLOR_BUFFER_BIT);
	double H = 0.20, W = 0.01;
	for(int i = step, w = 0; i < step + 200 && i < pattern.data.size(); i++, w++){
		for(int j = 0; j < 8; j++){
			double value = pattern.data[i][j];
			glColor3d(value, value, value);
			if(value > 1.0) printf("%d %d\n", i, j);
			glBegin(GL_QUADS);
			for(int t = 0; t < 4; t++){
				double x = H*(j+("0110"[t]-'0') * 0.5) - 1.0, y = W*(w+("0011"[t]-'0') * 0.5) - 1.0;
				glVertex2d(x, y);
			}
			glEnd();
		}
	}
	glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
	switch (key) {
		case 'a':     // ESC key
		step += 10;
		glutPostRedisplay();
	}
}

int main(int argc, char** argv) {
	glutInit(&argc, argv);
	glutCreateWindow("Vertex, Primitive & Color");
	glutInitDisplayMode(GLUT_DOUBLE);
	glutInitWindowSize(320, 320);
	glutInitWindowPosition(50, 50);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	pattern = pattern_parser("satellitebms/3|[Tanchiky] ENERGY SYNERGY MATRIX_another.bms", 0.1);
	glutMainLoop();
	return 0;
}

BOOST_PYTHON_MODULE(libpatternParser)
{
	np::initialize();
	p::def("patternParser", pattern_parser_python);
}