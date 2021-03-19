//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
#include "iostream"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0

	void main() {
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

float preferredDistance = 0.15;
float forceMultiplier = 1;
float friction = 0.05;

//Hyperbolic coordinates to normalized ones
// x/z, y/z
vec2 scaleToNormalized(vec3 hyperbolic) {
	return vec2(hyperbolic.x / hyperbolic.z, hyperbolic.y / hyperbolic.z);
}

//Normalized coordinates to hyperbolic ones
// 1 / (1-z^2)
vec3 backToHyperbolic(vec3 normalized) {
	return vec3(normalized.x, normalized.y, 1.0f) / sqrt(1.0f - (normalized.x) * (normalized.x) - (normalized.y) * (normalized.y));
}

const int numberOfNodes = 50;
const int visibleEdgesInPercent = 5;

class Node {
	vec3 position;
	float mass;
	vec3 speed;
	vec3 acceleration;
	unsigned int nodeVBO; //vbo for the nodes

public:

	Node() {
		randomizePosition();
		mass = 1;
		speed = vec3(0, 0, 0);
		acceleration = vec3(0, 0, 0);
	}

	Node(vec3 pos) {
		position = pos;
	}

	void randomizePosition() {
		// *2.0f is for the correct window size, the -1.0f is for the correct placement.
		//(without them the graph would be in the top-right corner)
		float x = (((float)rand() / RAND_MAX) * 2.0f) - 1.0f;
		float y = (((float)rand() / RAND_MAX) * 2.0f) - 1.0f;
		float z = 1;
		this->position = vec3(x, y, z);
	}

	vec3 getPosition() {
		return position;
	}

	void move(float deltaTime) {
		this->position = this->position + speed * deltaTime;
	}

	void applyForce(vec3 force, float deltaTime) { // TODO
		this->speed = this->speed + force / mass * deltaTime - friction * this->speed;
	}

	float getDistanceSqr(Node& other) {
		vec3 pos1 = this->getPosition();
		vec3 pos2 = other.getPosition();
		vec3 diff = pos1 - pos2;
		return dot(diff, diff);
	}

	float getDistance(Node& other) {
		return sqrtf(getDistanceSqr(other));
	}

	float getForceMagnitudeConnected(Node& other) {
		float distance = getDistance(other);
		if (distance <= preferredDistance) return -forceMultiplier * (distance - preferredDistance) * (distance - preferredDistance);
		else return forceMultiplier * (exp(distance - preferredDistance) - 1);
	}

	float getForceMagnitudeDisconnected(Node& other) {
		float distance = getDistance(other);
		return forceMultiplier * log(distance) - 1.5;
	}

	void draw() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &nodeVBO);
		glBindBuffer(GL_ARRAY_BUFFER, nodeVBO);

		std::vector<vec2> vertices;

		vertices.push_back(vec2(getPosition().x, getPosition().y));

		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), vertices.data(), GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		int location = glGetUniformLocation(gpuProgram.getId(), "color");
		glUniform3f(location, 1, 0, 0);
		float MVPtransf[4][4] = { 1, 0, 0, 0,
								  0, 1, 0, 0,
								  0, 0, 1, 0,
								  0, 0, 0, 1 };

		location = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

		glPointSize(10.0f);
		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, 1);
	}
};

class Edge {
	Node* n1;
	Node* n2;
	bool isVisible = false;
	vec3 color = vec3(0, 1, 0);
	unsigned int edgeVBO; //vbo for the edges

public:
	Edge(Node* node1, Node* node2, bool shouldDraw) {
		n1 = node1;
		n2 = node2;
		isVisible = shouldDraw;
	}

	void setVisible() {
		isVisible = true;
	}

	Node* getN1() {
		return n1;
	}

	Node* getN2() {
		return n2;
	}

	boolean getVisible() {
		return isVisible;
	}

	void draw() {
		if (isVisible) {
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			glGenBuffers(1, &edgeVBO);
			glBindBuffer(GL_ARRAY_BUFFER, edgeVBO);

			std::vector<vec2> vertices;

			vertices.push_back(vec2(n1->getPosition().x, n1->getPosition().y));
			vertices.push_back(vec2(n2->getPosition().x, n2->getPosition().y));

			glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(vec2), vertices.data(), GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

			int location = glGetUniformLocation(gpuProgram.getId(), "color");
			glUniform3f(location, color.x, color.y, color.z);
			float MVPtransf[4][4] = { 1, 0, 0, 0,
									  0, 1, 0, 0,
									  0, 0, 1, 0,
									  0, 0, 0, 1 };

			location = glGetUniformLocation(gpuProgram.getId(), "MVP");
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, vertices.size());
		}
	}
};

class Graph {
private:
	std::vector<Node> nodes;
	std::vector<Edge> edges;
public:
	void addNode(Node n) {
		nodes.push_back(n);
	}

	void addEdge(Edge e) {
		edges.push_back(e);
	}

	int requiredEdgeCount() {
		return (numberOfNodes * (numberOfNodes - 1) / 2) * visibleEdgesInPercent / 100;
	}

	std::vector<Node> getNodes() {
		return nodes;
	}

	std::vector<Edge> getEdges() {
		return edges;
	}

	void chooseVisibleEdges() {
		int requiredNumberOfVisibleEdges = requiredEdgeCount();
		int visibleEdgesCount = 0;
		while (visibleEdgesCount <= requiredNumberOfVisibleEdges) {
			int randomIndex = rand() % edges.size();
			if (edges[randomIndex].getVisible())
				continue;
			edges[randomIndex].setVisible();
			visibleEdgesCount++;
		}
	}

	void createGraph() {
		int visibleEdges = 0;

		for (size_t i = 0; i < nodes.size() - 1; i++)
		{
			for (size_t j = i + 1; j < nodes.size(); j++)
			{
				if (i == j)
					continue;
				addEdge(Edge(&nodes[i], &nodes[j], false));
			}
		}

		chooseVisibleEdges();

		printf("visibleEdges: %d", visibleEdges); // to check the number of visible/real edges
	}

	void draw() {
		for each (Node node in nodes)
		{
			node.draw();
		}
		for each (Edge edge in edges)
		{
			edge.draw();
		}
	}

	boolean isConnected(Node* n1, Node* n2) {
		for (int i = 0; i < edges.size(); i++)
		{
			if (edges[i].getN1() == n1 && edges[i].getN2() == n2 && edges[i].getVisible())
				return true;
		}
		return false;
	}

	vec3 calculateForceFrom(Node& current, Node& other) {
		vec3 diff = other.getPosition() - current.getPosition();
		//std::cout << current.getDistance(other) << std::endl;
		if (this->isConnected(&current, &other))
			return normalize(diff) * current.getForceMagnitudeConnected(other);
		return normalize(diff) * current.getForceMagnitudeDisconnected(other);
	}

	vec3 summariseForces(int nodeIndex) { //TODO
		vec3 sum = vec3(0, 0, 0);
		Node& currentNode = nodes[nodeIndex];
		for (int i = 0; i < nodes.size(); i++)
		{
			if (nodeIndex == i)
				continue;
			sum = sum + calculateForceFrom(currentNode, nodes[i]);
		}

		return sum;
	}

	void modifyGraph(float deltaTime) {
		//std::cout << "modify runs" << std::endl;
		for (int i = 0; i < nodes.size(); i++)
		{
			vec3 force = summariseForces(i);
			nodes[i].applyForce(force, deltaTime);
			nodes[i].move(deltaTime);
		}
	}
};

Graph graph;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	/*Node n1 = Node(vec3(0.0f, 0.0f, 0));
	Node n2 = Node(vec3(0.0f, 1.0f, 0));
	Node n3 = Node(vec3(1.0f, 0.0f, 0));
	Edge e1 = Edge(&n1, &n2, true);
	Edge e2 = Edge(&n1, &n3, true);
	Edge e3 = Edge(&n3, &n2, true);
	graph.addNode(n1);
	graph.addNode(n2);
	graph.addNode(n3);
	graph.addEdge(e1);

	std::cout<<graph.isConnected(&n1, &n2)<<std::endl;*/

	for (int i = 0; i < numberOfNodes; i++) {
		graph.addNode(Node());
	}
	graph.createGraph();

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	graph.draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program

	graph.modifyGraph(0.005);
	glutPostRedisplay();
}
