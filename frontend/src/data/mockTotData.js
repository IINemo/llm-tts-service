// Mock Tree-of-Thoughts data for testing
const mockTotData = {
  question: "What are 3 key benefits of microservices?",
  nodes: [
    {
      id: 0,
      step: 0,
      score: 0.0,
      state: "",
      is_root: true,
      is_selected: true,
      is_final: false
    },
    {
      id: 1,
      step: 1,
      score: 20.0,
      state: "**Break down the question**: Identify that we need 3 distinct benefits",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 2,
      step: 1,
      score: 1.0,
      state: "**Research existing knowledge**: Recall common microservices advantages",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 3,
      step: 1,
      score: 1.0,
      state: "**Analyze from architecture perspective**: Microservices promote modularity",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 4,
      step: 2,
      score: 20.0,
      state: "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 5,
      step: 2,
      score: 20.0,
      state: "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 6,
      step: 2,
      score: 1.0,
      state: "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services",
      is_root: false,
      is_selected: true,
      is_final: false
    },
    {
      id: 7,
      step: 3,
      score: 20.0,
      state: "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider scalability aspect**: Microservices can scale independently\n**Final answer**: The 3 key benefits are: 1) Independent scaling, 2) Technology flexibility, 3) Fault isolation",
      is_root: false,
      is_selected: true,
      is_final: true
    },
    {
      id: 8,
      step: 3,
      score: 20.0,
      state: "**Break down the question**: Identify that we need 3 distinct benefits\n**Consider development speed**: Teams can work independently on different services\n**Final answer**: The 3 key benefits are: 1) Faster development, 2) Independent deployment, 3) Team autonomy",
      is_root: false,
      is_selected: true,
      is_final: true
    },
    {
      id: 9,
      step: 3,
      score: 1.0,
      state: "**Analyze from architecture perspective**: Microservices promote modularity\n**Consider maintenance**: Easier to update individual services\n**Final answer**: The 3 key benefits are: 1) Modularity, 2) Easier updates, 3) Better code organization",
      is_root: false,
      is_selected: true,
      is_final: true
    }
  ],
  edges: [
    { from: 0, to: 1 },
    { from: 0, to: 2 },
    { from: 0, to: 3 },
    { from: 1, to: 4 },
    { from: 1, to: 5 },
    { from: 3, to: 6 },
    { from: 4, to: 7 },
    { from: 5, to: 8 },
    { from: 6, to: 9 }
  ],
  config: {
    beam_width: 3,
    steps: 3,
    method_generate: "propose"
  }
};

export default mockTotData;
