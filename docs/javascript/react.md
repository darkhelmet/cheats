# React Cheat Sheet

React is a JavaScript library for building user interfaces through reusable components. This cheat sheet covers React 18+ with modern hooks, patterns, and best practices.

## Quick Start

### Installation
```bash
# Create new React app
npx create-react-app my-app
cd my-app
npm start

# Or with Vite (recommended)
npm create vite@latest my-app -- --template react
cd my-app
npm install
npm run dev

# Add React to existing project
npm install react react-dom
```

### Basic Setup
```jsx
// index.js - Entry point
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
```

## Core Concepts

### JSX Fundamentals
```jsx
// JSX allows HTML-like syntax in JavaScript
function Welcome({ name }) {
  return (
    <div>
      <h1>Hello, {name}!</h1>
      <p>Welcome to React</p>
    </div>
  );
}

// JSX expressions
const user = { name: 'Alice', age: 30 };
const element = (
  <div>
    <p>Name: {user.name}</p>
    <p>Age: {user.age}</p>
    <p>Status: {user.age >= 18 ? 'Adult' : 'Minor'}</p>
  </div>
);

// JSX attributes
const imageUrl = 'https://example.com/image.jpg';
const image = <img src={imageUrl} alt="Description" className="image-style" />;
```

### Components

#### Function Components (Recommended)
```jsx
// Basic function component
function Greeting({ name }) {
  return <h1>Hello, {name}!</h1>;
}

// Arrow function component
const Greeting = ({ name }) => {
  return <h1>Hello, {name}!</h1>;
};

// Component with multiple props
function UserCard({ user, showEmail = false }) {
  return (
    <div className="user-card">
      <h2>{user.name}</h2>
      <p>Age: {user.age}</p>
      {showEmail && <p>Email: {user.email}</p>}
    </div>
  );
}
```

#### Class Components (Legacy)
```jsx
import React, { Component } from 'react';

class Welcome extends Component {
  render() {
    return <h1>Hello, {this.props.name}!</h1>;
  }
}
```

## State Management

### useState Hook
```jsx
import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  
  const increment = () => setCount(count + 1);
  const decrement = () => setCount(prev => prev - 1); // Functional update
  
  return (
    <div>
      <p>Count: {count}</p>
      <button onClick={increment}>+</button>
      <button onClick={decrement}>-</button>
    </div>
  );
}

// Multiple state variables
function UserProfile() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  return (
    <form>
      <input 
        value={name} 
        onChange={(e) => setName(e.target.value)}
        placeholder="Name"
      />
      <input 
        value={email} 
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
      />
    </form>
  );
}

// Object state
function UserSettings() {
  const [settings, setSettings] = useState({
    theme: 'light',
    notifications: true,
    language: 'en'
  });
  
  const updateTheme = (theme) => {
    setSettings(prev => ({ ...prev, theme }));
  };
  
  return (
    <div>
      <button onClick={() => updateTheme('dark')}>Dark Theme</button>
      <button onClick={() => updateTheme('light')}>Light Theme</button>
    </div>
  );
}
```

### useReducer Hook
```jsx
import { useReducer } from 'react';

// Reducer function
function counterReducer(state, action) {
  switch (action.type) {
    case 'INCREMENT':
      return { count: state.count + 1 };
    case 'DECREMENT':
      return { count: state.count - 1 };
    case 'RESET':
      return { count: 0 };
    default:
      throw new Error(`Unknown action: ${action.type}`);
  }
}

function Counter() {
  const [state, dispatch] = useReducer(counterReducer, { count: 0 });
  
  return (
    <div>
      <p>Count: {state.count}</p>
      <button onClick={() => dispatch({ type: 'INCREMENT' })}>+</button>
      <button onClick={() => dispatch({ type: 'DECREMENT' })}>-</button>
      <button onClick={() => dispatch({ type: 'RESET' })}>Reset</button>
    </div>
  );
}

// Complex state management
const todoReducer = (state, action) => {
  switch (action.type) {
    case 'ADD_TODO':
      return [...state, { id: Date.now(), text: action.text, done: false }];
    case 'TOGGLE_TODO':
      return state.map(todo => 
        todo.id === action.id ? { ...todo, done: !todo.done } : todo
      );
    case 'DELETE_TODO':
      return state.filter(todo => todo.id !== action.id);
    default:
      return state;
  }
};
```

## Effects and Side Effects

### useEffect Hook
```jsx
import { useEffect, useState } from 'react';

// Basic effect (runs after every render)
function Example() {
  const [count, setCount] = useState(0);
  
  useEffect(() => {
    document.title = `Count: ${count}`;
  });
  
  return <div>Count: {count}</div>;
}

// Effect with dependency array
function UserProfile({ userId }) {
  const [user, setUser] = useState(null);
  
  useEffect(() => {
    fetchUser(userId).then(setUser);
  }, [userId]); // Only re-run when userId changes
  
  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}

// Effect with cleanup
function Timer() {
  const [seconds, setSeconds] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setSeconds(prev => prev + 1);
    }, 1000);
    
    // Cleanup function
    return () => clearInterval(interval);
  }, []); // Empty dependency array = run once
  
  return <div>Seconds: {seconds}</div>;
}

// Multiple effects
function ChatRoom({ roomId }) {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  
  // Effect for connection
  useEffect(() => {
    const connection = createConnection(roomId);
    connection.connect();
    setIsConnected(true);
    
    return () => {
      connection.disconnect();
      setIsConnected(false);
    };
  }, [roomId]);
  
  // Effect for messages
  useEffect(() => {
    if (isConnected) {
      const unsubscribe = subscribeToMessages(roomId, setMessages);
      return unsubscribe;
    }
  }, [roomId, isConnected]);
}
```

## Event Handling

### Common Event Patterns
```jsx
function Form() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: ''
  });
  
  // Handle input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };
  
  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Form submitted:', formData);
    // Reset form
    setFormData({ name: '', email: '', message: '' });
  };
  
  // Handle button clicks
  const handleButtonClick = (e) => {
    console.log('Button clicked', e.target);
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        name="name"
        value={formData.name}
        onChange={handleChange}
        placeholder="Name"
      />
      <input
        type="email"
        name="email"
        value={formData.email}
        onChange={handleChange}
        placeholder="Email"
      />
      <textarea
        name="message"
        value={formData.message}
        onChange={handleChange}
        placeholder="Message"
      />
      <button type="submit">Submit</button>
      <button type="button" onClick={handleButtonClick}>Cancel</button>
    </form>
  );
}

// Event delegation and synthetic events
function ItemList({ items }) {
  const handleItemClick = (e, itemId) => {
    e.stopPropagation(); // Prevent event bubbling
    console.log(`Clicked item ${itemId}`);
  };
  
  return (
    <ul>
      {items.map(item => (
        <li key={item.id} onClick={(e) => handleItemClick(e, item.id)}>
          {item.name}
        </li>
      ))}
    </ul>
  );
}
```

## Props and Communication

### Props Patterns
```jsx
// Basic props
function Greeting({ name, age = 0 }) { // Default props
  return <p>Hello {name}, age {age}</p>;
}

// Destructuring props
function UserCard({ user: { name, email, avatar } }) {
  return (
    <div>
      <img src={avatar} alt={name} />
      <h3>{name}</h3>
      <p>{email}</p>
    </div>
  );
}

// Spread props
function Button({ children, ...props }) {
  return (
    <button {...props} className={`btn ${props.className || ''}`}>
      {children}
    </button>
  );
}

// Render props pattern
function DataFetcher({ render, url }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetch(url)
      .then(res => res.json())
      .then(data => {
        setData(data);
        setLoading(false);
      });
  }, [url]);
  
  return render({ data, loading });
}

// Usage
<DataFetcher 
  url="/api/users" 
  render={({ data, loading }) => 
    loading ? <div>Loading...</div> : <UserList users={data} />
  }
/>

// Children prop patterns
function Card({ children, title, footer }) {
  return (
    <div className="card">
      <h2>{title}</h2>
      <div className="card-content">{children}</div>
      {footer && <div className="card-footer">{footer}</div>}
    </div>
  );
}

// Higher-Order Component pattern
function withLoading(WrappedComponent) {
  return function WithLoadingComponent({ isLoading, ...props }) {
    if (isLoading) return <div>Loading...</div>;
    return <WrappedComponent {...props} />;
  };
}
```

### Lifting State Up
```jsx
function ParentComponent() {
  const [sharedState, setSharedState] = useState('');
  
  return (
    <div>
      <ChildA value={sharedState} onChange={setSharedState} />
      <ChildB value={sharedState} />
    </div>
  );
}

function ChildA({ value, onChange }) {
  return (
    <input 
      value={value} 
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function ChildB({ value }) {
  return <p>Current value: {value}</p>;
}
```

## Context API

### Creating and Using Context
```jsx
import { createContext, useContext, useState } from 'react';

// Create context
const ThemeContext = createContext();

// Provider component
function ThemeProvider({ children }) {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };
  
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

// Custom hook for using context
function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}

// Using context in components
function ThemedButton() {
  const { theme, toggleTheme } = useTheme();
  
  return (
    <button 
      className={`btn btn-${theme}`}
      onClick={toggleTheme}
    >
      Toggle Theme
    </button>
  );
}

// App with context
function App() {
  return (
    <ThemeProvider>
      <div className="app">
        <ThemedButton />
      </div>
    </ThemeProvider>
  );
}
```

### Complex Context with Reducer
```jsx
// Auth context with reducer
const AuthContext = createContext();

const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN':
      return { ...state, user: action.user, isAuthenticated: true };
    case 'LOGOUT':
      return { ...state, user: null, isAuthenticated: false };
    case 'SET_LOADING':
      return { ...state, isLoading: action.isLoading };
    default:
      return state;
  }
};

function AuthProvider({ children }) {
  const [state, dispatch] = useReducer(authReducer, {
    user: null,
    isAuthenticated: false,
    isLoading: true
  });
  
  const login = async (credentials) => {
    dispatch({ type: 'SET_LOADING', isLoading: true });
    try {
      const user = await authService.login(credentials);
      dispatch({ type: 'LOGIN', user });
    } catch (error) {
      console.error('Login failed:', error);
    } finally {
      dispatch({ type: 'SET_LOADING', isLoading: false });
    }
  };
  
  return (
    <AuthContext.Provider value={{ ...state, login, dispatch }}>
      {children}
    </AuthContext.Provider>
  );
}
```

## Custom Hooks

### Creating Custom Hooks
```jsx
// Custom hook for API calls
function useApi(url) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(url);
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [url]);
  
  return { data, loading, error };
}

// Custom hook for local storage
function useLocalStorage(key, initialValue) {
  const [storedValue, setStoredValue] = useState(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      return initialValue;
    }
  });
  
  const setValue = (value) => {
    try {
      setStoredValue(value);
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(error);
    }
  };
  
  return [storedValue, setValue];
}

// Custom hook for form handling
function useForm(initialValues) {
  const [values, setValues] = useState(initialValues);
  const [errors, setErrors] = useState({});
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setValues(prev => ({ ...prev, [name]: value }));
  };
  
  const handleSubmit = (callback) => (e) => {
    e.preventDefault();
    callback(values);
  };
  
  const reset = () => {
    setValues(initialValues);
    setErrors({});
  };
  
  return {
    values,
    errors,
    handleChange,
    handleSubmit,
    reset,
    setErrors
  };
}

// Usage
function ContactForm() {
  const { values, handleChange, handleSubmit, reset } = useForm({
    name: '',
    email: '',
    message: ''
  });
  
  const onSubmit = (formData) => {
    console.log('Submitting:', formData);
    reset();
  };
  
  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        name="name"
        value={values.name}
        onChange={handleChange}
        placeholder="Name"
      />
      {/* ... other inputs */}
    </form>
  );
}
```

## Performance Optimization

### React.memo
```jsx
import { memo } from 'react';

// Memoized component - only re-renders if props change
const ExpensiveComponent = memo(function ExpensiveComponent({ data, onUpdate }) {
  return (
    <div>
      {data.map(item => (
        <div key={item.id}>
          <span>{item.name}</span>
          <button onClick={() => onUpdate(item.id)}>Update</button>
        </div>
      ))}
    </div>
  );
});

// Custom comparison function
const OptimizedComponent = memo(function OptimizedComponent({ user, settings }) {
  return <div>{user.name} - {settings.theme}</div>;
}, (prevProps, nextProps) => {
  return prevProps.user.id === nextProps.user.id &&
         prevProps.settings.theme === nextProps.settings.theme;
});
```

### useMemo and useCallback
```jsx
import { useMemo, useCallback, useState } from 'react';

function ExpensiveList({ items, filter }) {
  // Memoize expensive calculations
  const filteredItems = useMemo(() => {
    return items.filter(item => item.name.includes(filter));
  }, [items, filter]);
  
  const expensiveValue = useMemo(() => {
    return items.reduce((sum, item) => sum + item.price, 0);
  }, [items]);
  
  // Memoize callback functions
  const handleItemClick = useCallback((itemId) => {
    console.log(`Clicked item ${itemId}`);
  }, []);
  
  const handleSort = useCallback((sortBy) => {
    // Sort logic here
  }, []);
  
  return (
    <div>
      <p>Total: ${expensiveValue}</p>
      {filteredItems.map(item => (
        <ItemCard
          key={item.id}
          item={item}
          onClick={handleItemClick}
        />
      ))}
    </div>
  );
}
```

## Conditional Rendering

### Conditional Patterns
```jsx
function ConditionalExample({ user, isLoggedIn, items = [] }) {
  return (
    <div>
      {/* Simple conditional */}
      {isLoggedIn && <p>Welcome, {user.name}!</p>}
      
      {/* Ternary operator */}
      {isLoggedIn ? (
        <UserDashboard user={user} />
      ) : (
        <LoginForm />
      )}
      
      {/* Complex conditions */}
      {isLoggedIn && user.role === 'admin' && (
        <AdminPanel />
      )}
      
      {/* Conditional with function */}
      {(() => {
        if (!isLoggedIn) return <LoginPrompt />;
        if (user.role === 'admin') return <AdminDashboard />;
        return <UserDashboard />;
      })()}
      
      {/* List rendering with conditions */}
      {items.length > 0 ? (
        <ul>
          {items.map(item => (
            <li key={item.id}>
              {item.name}
              {item.isNew && <span className="badge">New</span>}
            </li>
          ))}
        </ul>
      ) : (
        <p>No items found</p>
      )}
    </div>
  );
}
```

## Lists and Keys

### Rendering Lists
```jsx
function ItemList({ items, onDelete, onEdit }) {
  return (
    <div>
      {items.map(item => (
        <div key={item.id} className="item">
          <h3>{item.title}</h3>
          <p>{item.description}</p>
          <button onClick={() => onEdit(item)}>Edit</button>
          <button onClick={() => onDelete(item.id)}>Delete</button>
        </div>
      ))}
    </div>
  );
}

// Dynamic list with state
function TodoList() {
  const [todos, setTodos] = useState([]);
  const [newTodo, setNewTodo] = useState('');
  
  const addTodo = () => {
    if (newTodo.trim()) {
      setTodos(prev => [...prev, {
        id: Date.now(),
        text: newTodo,
        completed: false
      }]);
      setNewTodo('');
    }
  };
  
  const toggleTodo = (id) => {
    setTodos(prev => prev.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };
  
  return (
    <div>
      <div>
        <input
          value={newTodo}
          onChange={(e) => setNewTodo(e.target.value)}
          placeholder="Add todo..."
        />
        <button onClick={addTodo}>Add</button>
      </div>
      
      <ul>
        {todos.map(todo => (
          <li key={todo.id}>
            <span
              style={{
                textDecoration: todo.completed ? 'line-through' : 'none'
              }}
              onClick={() => toggleTodo(todo.id)}
            >
              {todo.text}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

## Forms and Validation

### Controlled Components
```jsx
function ContactForm() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    category: 'general',
    message: '',
    subscribe: false
  });
  
  const [errors, setErrors] = useState({});
  
  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };
  
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
    }
    
    if (!formData.message.trim()) {
      newErrors.message = 'Message is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      console.log('Form submitted:', formData);
      // Reset form
      setFormData({
        name: '',
        email: '',
        category: 'general',
        message: '',
        subscribe: false
      });
    }
  };
  
  return (
    <form onSubmit={handleSubmit}>
      <div>
        <label>
          Name:
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleInputChange}
          />
        </label>
        {errors.name && <span className="error">{errors.name}</span>}
      </div>
      
      <div>
        <label>
          Email:
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleInputChange}
          />
        </label>
        {errors.email && <span className="error">{errors.email}</span>}
      </div>
      
      <div>
        <label>
          Category:
          <select
            name="category"
            value={formData.category}
            onChange={handleInputChange}
          >
            <option value="general">General</option>
            <option value="support">Support</option>
            <option value="billing">Billing</option>
          </select>
        </label>
      </div>
      
      <div>
        <label>
          <input
            type="checkbox"
            name="subscribe"
            checked={formData.subscribe}
            onChange={handleInputChange}
          />
          Subscribe to newsletter
        </label>
      </div>
      
      <div>
        <label>
          Message:
          <textarea
            name="message"
            value={formData.message}
            onChange={handleInputChange}
            rows={4}
          />
        </label>
        {errors.message && <span className="error">{errors.message}</span>}
      </div>
      
      <button type="submit">Submit</button>
    </form>
  );
}
```

## Error Boundaries

### Error Boundary Component
```jsx
import { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }
  
  static getDerivedStateFromError(error) {
    return { hasError: true };
  }
  
  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }
  
  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <details>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }
    
    return this.props.children;
  }
}

// Usage
function App() {
  return (
    <ErrorBoundary>
      <Header />
      <MainContent />
      <Footer />
    </ErrorBoundary>
  );
}
```

## Refs and DOM Access

### useRef Hook
```jsx
import { useRef, useEffect } from 'react';

function FocusInput() {
  const inputRef = useRef(null);
  
  useEffect(() => {
    // Focus input when component mounts
    inputRef.current.focus();
  }, []);
  
  const handleFocus = () => {
    inputRef.current.focus();
  };
  
  return (
    <div>
      <input ref={inputRef} type="text" />
      <button onClick={handleFocus}>Focus Input</button>
    </div>
  );
}

// Refs with state
function VideoPlayer({ src }) {
  const videoRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  
  const togglePlayPause = () => {
    const video = videoRef.current;
    if (isPlaying) {
      video.pause();
    } else {
      video.play();
    }
    setIsPlaying(!isPlaying);
  };
  
  return (
    <div>
      <video ref={videoRef} src={src} />
      <button onClick={togglePlayPause}>
        {isPlaying ? 'Pause' : 'Play'}
      </button>
    </div>
  );
}

// Forwarding refs
import { forwardRef } from 'react';

const CustomInput = forwardRef(function CustomInput(props, ref) {
  return <input {...props} ref={ref} className="custom-input" />;
});

// Usage of forwarded ref
function Parent() {
  const inputRef = useRef(null);
  
  const focusInput = () => {
    inputRef.current.focus();
  };
  
  return (
    <div>
      <CustomInput ref={inputRef} />
      <button onClick={focusInput}>Focus</button>
    </div>
  );
}
```

## Common Patterns and Best Practices

### Component Composition
```jsx
// Composition over inheritance
function Layout({ children }) {
  return (
    <div className="layout">
      <Header />
      <main>{children}</main>
      <Footer />
    </div>
  );
}

// Compound components
function Tabs({ children, defaultTab = 0 }) {
  const [activeTab, setActiveTab] = useState(defaultTab);
  
  return (
    <div className="tabs">
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, { activeTab, setActiveTab, index })
      )}
    </div>
  );
}

function TabList({ children, activeTab, setActiveTab }) {
  return (
    <div className="tab-list">
      {React.Children.map(children, (child, index) =>
        React.cloneElement(child, {
          isActive: activeTab === index,
          onClick: () => setActiveTab(index)
        })
      )}
    </div>
  );
}

// Usage
<Tabs defaultTab={0}>
  <TabList>
    <Tab>Tab 1</Tab>
    <Tab>Tab 2</Tab>
  </TabList>
  <TabPanels>
    <TabPanel>Content 1</TabPanel>
    <TabPanel>Content 2</TabPanel>
  </TabPanels>
</Tabs>
```

### Data Fetching Patterns
```jsx
// Custom hook for data fetching with loading states
function useAsyncData(asyncFunction, dependencies = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    let isCancelled = false;
    
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await asyncFunction();
        if (!isCancelled) {
          setData(result);
        }
      } catch (err) {
        if (!isCancelled) {
          setError(err);
        }
      } finally {
        if (!isCancelled) {
          setLoading(false);
        }
      }
    };
    
    fetchData();
    
    return () => {
      isCancelled = true;
    };
  }, dependencies);
  
  return { data, loading, error };
}

// Usage
function UserProfile({ userId }) {
  const { data: user, loading, error } = useAsyncData(
    () => fetchUser(userId),
    [userId]
  );
  
  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!user) return <div>User not found</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
    </div>
  );
}
```

## Testing Patterns

### Component Testing Setup
```jsx
// Example test with React Testing Library
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Counter from './Counter';

test('renders counter and increments on click', async () => {
  render(<Counter />);
  
  // Find elements
  const countElement = screen.getByText(/count: 0/i);
  const incrementButton = screen.getByText(/increment/i);
  
  expect(countElement).toBeInTheDocument();
  
  // Simulate user interaction
  fireEvent.click(incrementButton);
  
  // Assert changes
  await waitFor(() => {
    expect(screen.getByText(/count: 1/i)).toBeInTheDocument();
  });
});

// Testing with context
test('uses theme context correctly', () => {
  render(
    <ThemeProvider>
      <ThemedButton />
    </ThemeProvider>
  );
  
  const button = screen.getByRole('button');
  expect(button).toHaveClass('btn-light');
  
  fireEvent.click(button);
  expect(button).toHaveClass('btn-dark');
});
```

## Common Gotchas and Best Practices

### Key Best Practices
```jsx
// ✅ Good: Stable keys for list items
items.map(item => <Item key={item.id} data={item} />)

// ❌ Bad: Using array index as key
items.map((item, index) => <Item key={index} data={item} />)

// ✅ Good: Functional state updates
setCount(prev => prev + 1);

// ❌ Bad: Direct state mutation
const newItems = items;
newItems.push(newItem);
setItems(newItems);

// ✅ Good: Immutable state updates
setItems(prev => [...prev, newItem]);

// ✅ Good: Effect cleanup
useEffect(() => {
  const subscription = subscribe();
  return () => subscription.unsubscribe();
}, []);

// ✅ Good: Dependency arrays
useEffect(() => {
  fetchData(userId);
}, [userId]); // Include all dependencies

// ❌ Bad: Missing dependencies
useEffect(() => {
  fetchData(userId);
}, []); // Missing userId dependency
```

### Performance Tips
```jsx
// Use React.memo for expensive components
const ExpensiveComponent = memo(({ data }) => {
  // Expensive rendering logic
  return <div>{/* Complex UI */}</div>;
});

// Use useMemo for expensive calculations
const expensiveValue = useMemo(() => {
  return data.reduce((acc, item) => acc + item.value, 0);
}, [data]);

// Use useCallback for stable function references
const handleClick = useCallback((id) => {
  onItemClick(id);
}, [onItemClick]);

// Lazy loading with React.lazy
const LazyComponent = lazy(() => import('./LazyComponent'));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <LazyComponent />
    </Suspense>
  );
}
```

This cheat sheet covers the essential React concepts, patterns, and best practices for building modern React applications. Focus on understanding hooks, component patterns, and state management for effective React development.