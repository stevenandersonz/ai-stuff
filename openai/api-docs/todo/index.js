const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");

const app = express();
const Todos = [
  {
    id: 1,
    title: "Buy groceries",
    description: "Milk, eggs, bread",
    completed: false,
  },
  {
    id: 2,
    title: "Clean the house",
    description: "Vacuum, dust, mop",
    completed: false,
  },
  {
    id: 3,
    title: "Walk the dog",
    description: "Take Fido for a stroll",
    completed: true,
  },
  {
    id: 4,
    title: "Go to the gym",
    description: "Cardio and weights",
    completed: false,
  },
  {
    id: 5,
    title: "Read a book",
    description: "Choose something from the library",
    completed: true,
  },
];
app.use(cors());
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());

// <START>
app.get("/todos", (req, res) => {
  res.send(Todos);
});
// <END>

// <START>
app.post("/todos", (req, res) => {
  const todo = {
    title: req.body.title,
    description: req.body.description,
    completed: false,
  };
  Todos.push(todo);
  res.send(todo);
});
// <END>

// <START>
app.get("/todos/:id", (req, res) => {
  let todo = Todos.find((todo) => todo.id === parseInt(req.params.id));
  res.send(todo);
});
// <END>

// <START>
app.put("/todos/:id", (req, res) => {
  let todo = Todos.find((todo) => todo.id === parseInt(req.params.id));
  todo.title = req.body.title;
  todo.description = req.body.description;
  todo.completed = req.body.completed;
  for (let i = 0; i < Todos.length; i++) {
    if (Todos[i].id === todo.id) {
      Todos[i] = todo;
    }
  }
  res.send(todo);
});
// <END>

app.listen(3000, () => {
  console.log("Server is running on port 3000");
});
