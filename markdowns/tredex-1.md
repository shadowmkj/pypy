## Software Requirements Specification (SRS)

## for Tredex - Online Sneakers/Shoes Marketplace

Version 2.0

Dated: 06/09/2025

## Table of Contents

1. Introduction
2. 1.1. Purpose
3. 1.2. Document Conventions
4. 1.3. Scope
5. 1.4. Definitions, Acronyms, and Abbreviations
2. Overall Description
7. 2.1. Product Perspective
8. 2.2. Product Features
9. 2.3. User Classes and Characteristics
10. 2.4. Operating Environment
11. 2.5. Design and Implementation Constraints
3. System Features (Functional Requirements)
13. 3.1. Landing Page
14. 3.2. Product Catalog and Viewing
15. 3.3. Product Filtering and Search
16. 3.4. Detailed Product View
17. 3.5. Ordering via WhatsApp
18. 3.6. Admin Panel
4. External Interface Requirements
20. 4.1. User Interfaces
21. 4.2. Software Interfaces
22. 4.3. Hardware Interfaces
23. 4.4. Communications Interfaces
5. Non-Functional Requirements
25. 5.1. Performance Requirements
26. 5.2. Usability Requirements
27. 5.3. Reliability
28. 5.4. Security Requirements

## 1. Introduction

## 1.1. Purpose

This document provides a detailed description of the requirements for the Tredex e-commerce web application. Its purpose is to serve as a foundational guide for developers, testers, and project stakeholders, ensuring that the final product meets the specified needs. Tredex will function as an online marketplace for customers to browse, view, and order shoes.

## 1.2. Document Conventions

This document follows a standard SRS template. The system shall be referred to as "Tredex" or "the application".

## 1.3. Scope

The scope of this project is to develop a responsive web application that allows users to:

- View a home page with featured products and categories.
- Browse a comprehensive product catalog of shoes.
- Filter products by brand, category, and price.
- View detailed information and images for each specific shoe.
- Initiate an order by sending a pre-formatted message through the WhatsApp API to a designated business number.
- Provide a secure admin panel for administrators to perform Create, Read, Update, and Delete (CRUD) operations on products and categories.

This version will not include customer accounts, online payment processing, or a shopping cart feature. The database will only store product and category data, not customer or order data. All order fulfillment will be handled externally after the WhatsApp message is received.

## 1.4. Definitions, Acronyms, and Abbreviations

- SRS: Software Requirements Specification
- ●
- UI: User Interface
- API: Application Programming Interface
- SKU: Stock Keeping Unit. A unique identifier for each distinct product.
- CRUD:

Create, Read, Update, Delete.

## 2. Overall Description

## 2.1. Product Perspective

Tredex is a new, standalone web application. It will serve as the primary online storefront for the business, using a PostgreSQL database for storing and managing all product and category information. Its main purpose for customers is to generate sales leads through WhatsApp. The application will be self-contained but will rely on an external WhatsApp Business account for its core ordering functionality.

## 2.2. Product Features

The major features of the Tredex web application are:

- Landing Page: An attractive entry point for visitors.
- Product Catalog: A multi-page display of all available shoes.
- Brand-Based Catalog: A way to view all shoes from a specific brand.
- Categorization: Products organized into logical categories.
- Price-Based Filtering: A simple mechanism to narrow down products based on price.
- Detailed Product Pages: Individual pages for each product with comprehensive details.
- WhatsApp Order Integration: A "Buy Now" button that launches WhatsApp with pre-filled order details.
- Admin Panel: A secure area for administrators to log in and manage products and categories.

## 2.3. User Classes and Characteristics

There are two primary classes of users for Tredex:

- Customers: General web users looking to purchase shoes. They may have varying levels of technical proficiency but are expected to be familiar with standard e-commerce browsing and mobile messaging apps like WhatsApp.
- Administrators: A technical user responsible for managing the website's content. This user will have credentials to log in to a protected admin area to add, update, and remove products and categories.

## 2.4. Operating Environment

The application shall be a server-side rendered web application built using the Next.js framework . It will be hosted on a Node.js compatible environment. A PostgreSQL database will be used as the data store for all product and category information. The application must be accessible through modern web browsers (e.g., Chrome, Firefox, Safari, Edge).

## 2.5. Design and Implementation Constraints

- The application must be responsive and provide a seamless experience on mobile, tablet, and desktop screens.
- Order processing is strictly limited to the WhatsApp API. No other ordering methods or databases for orders will be implemented.
- The user interface should be clean, modern, and visually appealing, with a strong focus on product imagery.

## 3. System Features (Functional Requirements)

## 3.1. Landing Page

- 3.1.1. Hero Section: The page shall display a prominent "hero" banner or image carousel to showcase featured collections or promotions.
- 3.1.2. Featured Products: A section shall display a curated list of "Featured" or "New Arrival" products with their name, image, and price.
- 3.1.3. Category Links: The page shall display links or visual blocks representing the
- main product categories.
- 3.1.4. Brand Showcase: A section shall display logos of the key brands available on the platform, which link to their respective brand pages.
- 3.1.5. Navigation: A persistent navigation bar shall be present at the top of the page with links to Home, Categories, and Brands.
- 3.1.6. Footer: A footer shall be present with links to informational pages (About Us, Contact, T&amp;C) and social media profiles.

## 3.2. Product Catalog and Viewing

- 3.2.1. Grid View: Products in the catalog shall be displayed in a responsive grid layout. Each item in the grid will show the primary product image, product name, brand, and price.
- 3.2.2. Pagination: If the number of products exceeds a set limit per page, pagination controls shall be available.
- 3.2.3. Category Catalog: Users shall be able to navigate to a specific category page that displays only the products within that category.
- 3.2.4. Brand Catalog: Users shall be able to navigate to a specific brand page that displays only the products from that brand.

## 3.3. Product Filtering

- 3.3.1. Price Filter: The catalog pages shall include a simple price filtering mechanism, such as checkboxes for predefined ranges or a price slider.
- 3.3.2. Filter Application: Applying a filter shall dynamically update the product grid on the page without a full page reload.

## 3.4. Detailed Product View

- 3.4.1. Product Images: The page shall display multiple high-resolution images of the product.
- 3.4.2. Product Information: The page shall display the Product Name, Brand Name, Price, Detailed Description, SKU, and Available Materials/Colors.
- 3.4.3. Size Selection: A dropdown menu or selectable buttons shall be provided to choose a shoe size. The "Order" button should be disabled until a size is selected.

## 3.5. Ordering via WhatsApp

- 3.5.1. Order Button: A clearly visible button labeled "Order on WhatsApp" shall be present on the product page.
- 3.5.2. WhatsApp API Integration: Clicking the button shall trigger a redirect to the WhatsApp API (wa.me/ link).
- 3.5.3. Pre-formatted Message: The WhatsApp chat window should open with a pre-filled message. The message template shall be:

Hi, I would like to order the following product from Tredex:

Product: [Product Name]

Brand: [Brand Name]

Size: [Selected Size]

Price: [Product Price]

Link:

[https://www.shopify.com/blog/product-page](https://www.shopify.com/blog/product-p age)

- 3.5.4. Message Trigger: The user must manually press "send" within their WhatsApp application to transmit the order message.

## 3.6. Admin Panel

- 3.6.1. Admin Authentication:
- The system shall provide a secure login page (e.g., at /admin/login) for administrators.
- Access to any other /admin/* routes shall be protected and require a valid, authenticated session.
- 3.6.2. Product Management (CRUD):
- Create: The admin shall be able to access a form to add a new product. The form must include fields for name, description, price, SKU, brand, available sizes, and image uploads.
- Read: The admin panel shall display a paginated list of all products in a table, showing key details and providing options to edit or delete each one.
- Update: The admin shall be able to edit the details of any existing product through a pre-filled form.
- Delete: The admin shall be able to delete a product from the database. A confirmation prompt shall be displayed before deletion.
- 3.6.3. Category Management (CRUD):
- Create: The admin shall be able to add a new product category (e.g., "Boots", "Sneakers").
- Read: The admin panel shall display a list of all categories.
- Update: The admin shall be able to rename an existing category.
- Delete: The admin shall be able to delete a category. The system should prevent the deletion of a category if it is currently assigned to any products.

## 4. External Interface Requirements

## 4.1. User Interfaces

- The UI shall be clean, intuitive, and modern.
- The design shall be fully responsive.
- The admin panel UI should be functional and straightforward, prioritizing ease of data management.

## 4.2. Software Interfaces

- WhatsApp Business API: The system will interface with the public WhatsApp "click to chat" feature using a formatted wa.me/ URL.
- Database Interface: The Next.js application will interface with a PostgreSQL database via a database client or ORM (e.g., Prisma, Drizzle).

## 4.3. Hardware Interfaces

No specific hardware interfaces are required.

## 4.4. Communications Interfaces

The application requires an active internet connection on the user's device.

## 5. Non-Functional Requirements

## 5.1. Performance Requirements

- Web pages should load quickly, with a target load time of under 4 seconds on a standard broadband connection.
- API responses from the Next.js backend should be processed in under 1200ms.
- The application should handle at least 50 concurrent users without significant degradation in performance.

## 5.2. Usability Requirements

- The customer-facing navigation shall be straightforward.
- The admin panel shall be intuitive for a technical user to manage products and categories with minimal training.

## 5.3. Reliability

The website shall have an uptime of at least 99.5%.

## 5.4. Security Requirements

- All communication shall be encrypted using HTTPS (SSL/TLS).
- Admin passwords must be securely hashed and salted before being stored in the database.
- Standard precautions against common web vulnerabilities (e.g., XSS, CSRF, SQL Injection) must be implemented.