# Protocol Buffers (protobuf) Cheat Sheet

Protocol Buffers is Google's language-neutral, platform-neutral mechanism for serializing structured data. This guide covers .proto file syntax and compilation with `protoc`.

## Quick Start

### Installation
```bash
# Install protoc compiler (Linux/Mac)
# From https://github.com/protocolbuffers/protobuf/releases
curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v27.0/protoc-27.0-linux-x86_64.zip
unzip protoc-27.0-linux-x86_64.zip -d protoc
sudo cp protoc/bin/protoc /usr/local/bin/
sudo cp -r protoc/include/google /usr/local/include/

# Verify installation
protoc --version
```

### Language Plugins
```bash
# Go
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest

# Python (included with protobuf library)
pip install protobuf

# Java/Kotlin (built into protoc)
# JavaScript/TypeScript
npm install -g @protobuf-ts/plugin
```

## Core Concepts

### .proto File Structure
```proto
// File header
syntax = "proto3";  // or "proto2" or edition = "2024"
package com.example;

// Imports
import "google/protobuf/timestamp.proto";

// Options
option java_package = "com.example.proto";
option go_package = "./proto";

// Message definitions
message Person {
  string name = 1;
  int32 age = 2;
}
```

## Message Definitions

### Basic Message
```proto
syntax = "proto3";

message User {
  int32 id = 1;           // Field number must be unique
  string username = 2;
  string email = 3;
  bool active = 4;
}
```

### Nested Messages
```proto
message Person {
  string name = 1;
  
  message Address {
    string street = 1;
    string city = 2;
    string country = 3;
  }
  
  Address address = 2;
}
```

### Field Rules (Proto2 vs Proto3)
```proto
// Proto2
syntax = "proto2";
message User {
  required string name = 1;    // Must be provided
  optional int32 age = 2;      // Can be omitted
  repeated string tags = 3;    // Can have 0 or more values
}

// Proto3 (more common)
syntax = "proto3";
message User {
  string name = 1;             // Implicit presence
  optional int32 age = 2;      // Explicit presence (proto3.15+)
  repeated string tags = 3;    // 0 or more values
}
```

## Scalar Types

### Numeric Types
```proto
message NumericTypes {
  double price = 1;      // 64-bit floating point
  float rating = 2;      // 32-bit floating point
  
  int32 count = 3;       // Variable-length encoding
  int64 big_count = 4;   // Variable-length encoding
  
  uint32 positive = 5;   // Variable-length encoding
  uint64 big_positive = 6;
  
  sint32 signed_val = 7;  // Better for negative numbers
  sint64 big_signed = 8;
  
  fixed32 fixed_val = 9;  // Always 4 bytes
  fixed64 big_fixed = 10; // Always 8 bytes
  
  sfixed32 signed_fixed = 11; // Always 4 bytes, signed
  sfixed64 big_signed_fixed = 12; // Always 8 bytes, signed
  
  bool enabled = 13;
}
```

### String and Bytes
```proto
message TextTypes {
  string text = 1;       // UTF-8 encoded string
  bytes data = 2;        // Arbitrary byte sequence
}
```

## Complex Types

### Enumerations
```proto
enum Status {
  UNKNOWN = 0;    // First value must be 0 in proto3
  PENDING = 1;
  APPROVED = 2;
  REJECTED = 3;
}

message Request {
  Status status = 1;
}
```

### Repeated Fields
```proto
message ShoppingCart {
  repeated string items = 1;              // List of strings
  repeated int32 quantities = 2;          // List of integers
  repeated Product products = 3;          // List of messages
}

// With packed encoding (more efficient for primitives)
message Measurements {
  repeated int32 values = 1 [packed = true];
}
```

### Maps
```proto
message UserSettings {
  map<string, string> preferences = 1;    // String to string
  map<int32, User> users_by_id = 2;      // Integer to message
  map<string, bool> features = 3;        // String to boolean
}
```

### Oneof (Union Types)
```proto
message SearchRequest {
  string query = 1;
  
  oneof filter {
    string category = 2;
    int32 price_max = 3;
    bool on_sale = 4;
  }
}
```

## Advanced Features

### Any Type
```proto
import "google/protobuf/any.proto";

message ErrorInfo {
  string message = 1;
  google.protobuf.Any details = 2;  // Can contain any message type
}
```

### Well-Known Types
```proto
import "google/protobuf/timestamp.proto";
import "google/protobuf/duration.proto";
import "google/protobuf/empty.proto";
import "google/protobuf/struct.proto";

message Event {
  google.protobuf.Timestamp created_at = 1;
  google.protobuf.Duration timeout = 2;
  google.protobuf.Struct metadata = 3;
}
```

### Field Options
```proto
message Product {
  string name = 1;
  double price = 2 [(validate.rules).double.gt = 0];  // Custom validation
  string description = 3 [deprecated = true];         // Mark as deprecated
}
```

### Reserved Fields
```proto
message User {
  reserved 2, 15, 9 to 11;           // Reserve field numbers
  reserved "old_name", "legacy_id";  // Reserve field names
  
  string name = 1;
  string email = 3;
  // Field 2 is reserved and cannot be used
}
```

## Services (gRPC)

### Basic Service Definition
```proto
syntax = "proto3";

service UserService {
  // Unary RPC
  rpc GetUser(GetUserRequest) returns (User);
  
  // Server streaming
  rpc ListUsers(ListUsersRequest) returns (stream User);
  
  // Client streaming
  rpc CreateUsers(stream CreateUserRequest) returns (CreateUsersResponse);
  
  // Bidirectional streaming
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}

message GetUserRequest {
  int32 user_id = 1;
}

message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
}

message CreateUserRequest {
  User user = 1;
}

message CreateUsersResponse {
  repeated User users = 1;
}

message ChatMessage {
  string message = 1;
  string user_id = 2;
}
```

## Compilation with protoc

### Basic Usage
```bash
# Check version
protoc --version

# Generate code for single language
protoc --python_out=./generated user.proto

# Generate for multiple languages
protoc --java_out=./java --python_out=./python --go_out=./go user.proto

# Specify import paths
protoc -I./protos -I./vendor --python_out=./generated protos/user.proto
```

### Language-Specific Generation

#### Go
```bash
# Basic Go generation
protoc --go_out=. --go_opt=paths=source_relative user.proto

# With gRPC
protoc --go_out=. --go-grpc_out=. \
  --go_opt=paths=source_relative \
  --go-grpc_opt=paths=source_relative \
  user.proto

# Custom Go package
protoc --go_out=. --go_opt=Muser.proto=./internal/proto user.proto
```

#### Python
```bash
# Basic Python generation
protoc --python_out=./generated user.proto

# With type stubs (.pyi files)
protoc --python_out=./generated --pyi_out=./generated user.proto

# With gRPC
protoc --python_out=./generated --grpc_python_out=./generated user.proto
```

#### Java/Kotlin
```bash
# Java
protoc --java_out=./src/main/java user.proto

# Java Lite (smaller runtime)
protoc --java_out=lite:./src/main/java user.proto

# Kotlin
protoc --java_out=./src/main/java --kotlin_out=./src/main/kotlin user.proto
```

#### JavaScript/TypeScript
```bash
# JavaScript
protoc --js_out=import_style=commonjs,binary:./generated user.proto

# TypeScript with protobuf-ts
protoc --ts_out=./generated user.proto
```

#### C++
```bash
# C++
protoc --cpp_out=./generated user.proto

# With gRPC
protoc --cpp_out=./generated --grpc_out=./generated \
  --plugin=protoc-gen-grpc=grpc_cpp_plugin user.proto
```

#### C#
```bash
# C#
protoc --csharp_out=./Generated user.proto

# With gRPC
protoc --csharp_out=./Generated --grpc_out=./Generated \
  --plugin=protoc-gen-grpc=grpc_csharp_plugin user.proto
```

### Advanced protoc Options

#### Multiple Files and Directories
```bash
# Compile all .proto files in directory
protoc --python_out=./generated protos/*.proto

# Recursive compilation
find ./protos -name "*.proto" -exec protoc --python_out=./generated {} \;

# With include paths
protoc -I./protos -I./vendor -I./third_party \
  --python_out=./generated \
  protos/user.proto protos/order.proto
```

#### Descriptor Sets
```bash
# Generate descriptor set (for reflection)
protoc --descriptor_set_out=user.desc --include_imports user.proto

# Generate descriptor set with source info
protoc --descriptor_set_out=user.desc \
  --include_imports --include_source_info user.proto
```

#### Custom Plugins
```bash
# Use custom plugin
protoc --plugin=protoc-gen-custom=./my-plugin \
  --custom_out=./generated user.proto

# Plugin with options
protoc --plugin=protoc-gen-validate=protoc-gen-validate \
  --validate_out="lang=go:./generated" user.proto
```

## Build Integration

### Makefile
```makefile
# Variables
PROTO_FILES = $(wildcard protos/*.proto)
GENERATED_GO = $(PROTO_FILES:protos/%.proto=generated/%.pb.go)
GENERATED_PY = $(PROTO_FILES:protos/%.proto=generated/%_pb2.py)

# Go generation
generated/%.pb.go: protos/%.proto
	protoc --go_out=generated --go_opt=paths=source_relative $<

# Python generation
generated/%_pb2.py: protos/%.proto
	protoc --python_out=generated $<

# Targets
.PHONY: go python clean
go: $(GENERATED_GO)
python: $(GENERATED_PY)

clean:
	rm -rf generated/*

all: go python
```

### CMake
```cmake
# Find protobuf
find_package(protobuf REQUIRED)

# Function to compile protobuf files
function(compile_proto_files)
    foreach(proto_file ${ARGN})
        get_filename_component(proto_name ${proto_file} NAME_WE)
        get_filename_component(proto_dir ${proto_file} DIRECTORY)
        
        set(generated_files
            ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.h
            ${CMAKE_CURRENT_BINARY_DIR}/${proto_name}.pb.cc
        )
        
        add_custom_command(
            OUTPUT ${generated_files}
            COMMAND protobuf::protoc
            ARGS --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                 -I${proto_dir}
                 ${proto_file}
            DEPENDS ${proto_file}
        )
        
        list(APPEND PROTO_GENERATED_FILES ${generated_files})
    endforeach()
    
    set(PROTO_GENERATED_FILES ${PROTO_GENERATED_FILES} PARENT_SCOPE)
endfunction()

# Usage
compile_proto_files(protos/user.proto protos/order.proto)
add_executable(myapp main.cpp ${PROTO_GENERATED_FILES})
target_link_libraries(myapp protobuf::libprotobuf)
```

### Bazel
```python
# BUILD file
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@io_grpc_grpc_java//:java_grpc_library.bzl", "java_grpc_library")

proto_library(
    name = "user_proto",
    srcs = ["user.proto"],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
    ],
)

java_proto_library(
    name = "user_java_proto",
    deps = [":user_proto"],
)

java_grpc_library(
    name = "user_java_grpc",
    srcs = [":user_proto"],
    deps = [":user_java_proto"],
)
```

## Best Practices

### Schema Design
```proto
// Good: Use clear, descriptive names
message UserProfile {
  string full_name = 1;        // Better than 'name'
  string email_address = 2;    // Better than 'email'
  int64 created_timestamp = 3; // Better than 'created'
}

// Good: Group related fields
message Address {
  string street_line_1 = 1;
  string street_line_2 = 2;
  string city = 3;
  string state = 4;
  string postal_code = 5;
  string country_code = 6;
}

// Good: Use enums for fixed sets of values
enum UserRole {
  ROLE_UNSPECIFIED = 0;  // Always include zero value
  ROLE_USER = 1;
  ROLE_ADMIN = 2;
  ROLE_MODERATOR = 3;
}
```

### Field Numbering
```proto
message Product {
  // Reserve low numbers (1-15) for frequently used fields
  // They use 1 byte for tag encoding
  string name = 1;
  double price = 2;
  bool available = 3;
  
  // Higher numbers (16+) use 2+ bytes
  string detailed_description = 16;
  repeated string tags = 17;
  
  // Reserve ranges for future use
  reserved 4 to 10;
  reserved 100 to 200;
}
```

### Versioning and Evolution
```proto
// Original version
message User {
  string name = 1;
  string email = 2;
}

// Evolved version - backward compatible
message User {
  string name = 1;
  string email = 2;
  
  // New optional fields don't break compatibility
  optional int32 age = 3;
  repeated string interests = 4;
  
  // Nested messages can be added
  optional Address address = 5;
}
```

### Performance Considerations
```proto
// Use appropriate field types for your data
message Metrics {
  // Use packed repeated for primitive arrays
  repeated int32 values = 1 [packed = true];
  
  // Consider fixed types for known-size data
  fixed64 timestamp_nanos = 2;  // Better than int64 for large numbers
  
  // Use bytes for binary data
  bytes thumbnail = 3;          // Not string for binary data
  
  // Consider string vs bytes for text
  string utf8_text = 4;         // For valid UTF-8
  bytes raw_text = 5;           // For potentially invalid UTF-8
}
```

## Common Patterns

### Request/Response Patterns
```proto
// Standard CRUD operations
service UserService {
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
}

message CreateUserRequest {
  User user = 1;
}

message CreateUserResponse {
  User user = 1;
  string message = 2;
}

message GetUserRequest {
  string user_id = 1;
}

message GetUserResponse {
  User user = 1;
}

// Pagination pattern
message ListUsersRequest {
  int32 page_size = 1;
  string page_token = 2;
  string filter = 3;
}

message ListUsersResponse {
  repeated User users = 1;
  string next_page_token = 2;
  int32 total_count = 3;
}
```

### Error Handling
```proto
import "google/rpc/status.proto";
import "google/protobuf/any.proto";

message ErrorResponse {
  google.rpc.Status status = 1;  // Standard error status
  string message = 2;            // Human-readable message
  repeated google.protobuf.Any details = 3;  // Additional error details
}

// Custom error details
message ValidationError {
  repeated FieldError field_errors = 1;
}

message FieldError {
  string field = 1;
  string message = 2;
  string code = 3;
}
```

## Gotchas and Common Mistakes

### Field Number Management
```proto
// DON'T: Reuse field numbers
message User {
  string name = 1;
  // string old_email = 2;  // Removed field
  string email = 2;         // DON'T reuse number 2
}

// DO: Reserve removed field numbers
message User {
  reserved 2;  // or: reserved "old_email";
  string name = 1;
  string email = 3;  // Use new number
}
```

### Default Values and Presence
```proto
// Proto3: Cannot distinguish between default value and not set
message User {
  int32 age = 1;  // age=0 could mean "not set" or actually 0
}

// Solution: Use optional or wrapper types
import "google/protobuf/wrappers.proto";

message User {
  optional int32 age = 1;  // Can detect presence
  // or
  google.protobuf.Int32Value age_wrapper = 2;
}
```

### Package and Import Issues
```proto
// File: protos/user.proto
syntax = "proto3";
package myapp.user;  // Use consistent package naming

import "protos/common.proto";  // Use relative paths consistently

// File: protos/common.proto
syntax = "proto3";
package myapp.common;  // Must match directory structure
```

### Compilation Ordering
```bash
# Wrong: May fail if dependencies aren't found
protoc --python_out=. user.proto

# Right: Include all necessary import paths
protoc -I. -I./vendor -I./third_party --python_out=. user.proto
```

## Quick Reference

### Essential protoc Flags
| Flag | Purpose | Example |
|------|---------|---------|
| `--version` | Show protoc version | `protoc --version` |
| `-I, --proto_path` | Add import directory | `protoc -I./protos` |
| `--python_out` | Generate Python code | `--python_out=./gen` |
| `--go_out` | Generate Go code | `--go_out=.` |
| `--java_out` | Generate Java code | `--java_out=./src` |
| `--cpp_out` | Generate C++ code | `--cpp_out=./gen` |
| `--descriptor_set_out` | Generate descriptor | `--descriptor_set_out=desc.pb` |
| `--include_imports` | Include dependencies in descriptor | Use with `--descriptor_set_out` |

### Scalar Type Mapping
| Proto Type | Go | Python | Java | C++ | JavaScript |
|------------|----|---------|----- |----|------------|
| `double` | `float64` | `float` | `double` | `double` | `number` |
| `float` | `float32` | `float` | `float` | `float` | `number` |
| `int32` | `int32` | `int` | `int` | `int32` | `number` |
| `int64` | `int64` | `int` | `long` | `int64` | `string` |
| `string` | `string` | `str` | `String` | `string` | `string` |
| `bool` | `bool` | `bool` | `boolean` | `bool` | `boolean` |
| `bytes` | `[]byte` | `bytes` | `ByteString` | `string` | `Uint8Array` |

---

*Protocol Buffers provide efficient, language-agnostic data serialization with strong schema evolution capabilities. Focus on clear field naming, proper type selection, and maintaining backward compatibility for production systems.*