---
layout: post
title: "DNS Explained"
---


Hi! Welcome to this blog post about the Domain Name System (DNS). I decided to write about this since I attended a cyber security course last week and found this particularly interesting.

### What is DNS?

For browsers to access content of a website, they need to know the IP address of the website's IP address. An IP address could looking something like 192.168.1.1.

However, it is difficult for humans to remember full IP addresses, so domain names are used instead e.g google.com and bbc.co.uk

DNS is responsible for translating domain names into IP addresses, so that browsers can access internet resources when given a domain name. 

### How does DNS work?

There are 4 types of DNS servers: DNS recursor, Root nameserver, TLD nameserver and the Authoritative name server.

- The **DNS recursor** is responsible for taking requests from the client. It would then relay the request to other DNS servers in order to satisfy the request.

- The **root nameserver** looks at the top level domain of the requested domain name and directs the request to the relevant TLD nameserver.

- The **TLD (Top Level Domain) nameserver** hosts the last part of a domain name (eg "com" in google.com). It then directs the request to an authoratative nameserver.

- The **authoritative nameserver** holds the actual record of the requested domain name's IP address. 

**Steps of DNS lookup**

1. User types "example.com" into a browser and the request is received by the DNS recursor
2. The resolver queries a root nameserver
3. The root server returns the address of the relevant TLD (Top Level Domain) nameserver. In this case, the TLD server would be the .com server.
4. The resolver makes a request to the TLD server
5. The TLD server responds with the IP address of the domain's authoritative nameserver 
6. The recursive resolver then query's the domain's nameserver
7. The nameserver returns the domain name's IP address
8. The resolver comes back to the browser with the IP address, which now allows the browser to load the website's content


### DNS Caching

DNS Caching makes resolving domain names much quicker, by storing earlier resolved domain names close to the client. 

**Browser DNS Caching**

Modern web browsers often cache DNS records for a certain amount of time. This means less steps are needed to resolve a domain name. The browser cache is the first thing checked when a DNS record request is made. 

**OS Level DNS Caching**

The operating system's DNS resolver is the last local stop before the DNS query leaves the machine. The process in the OS which handles this request is a "stub resolver". 


A stub resolver checks its own cache to see if it has the record. If it doesn't, it then sends a DNS query to a DNS recursive resolver in the ISP (internet service provider)

The ISP's recursive resolver would then check if it has the record locally. 

The recursive resolver can also do the following, depending on what it has cached:

- If the resolver has the nameserver record for the authoritative nameserver, but not the address record, it will query the authoritative nameserver directly, which skips a lot of steps in the normal DNS query process.

- If it does not have the nameserver record, it skips the root server and queries the TLD server directly.

- If there are no records pointing to a TLD server, the resolver queries the root servers.
